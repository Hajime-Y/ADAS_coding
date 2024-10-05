import argparse
import copy
import json
import os
import random
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import backoff
import numpy as np
import openai
import pandas
from tqdm import tqdm

from coding_prompt import get_init_archive, get_prompt, get_reflexion_prompt, get_evaluation_prompt

# .envファイルの内容を読み込見込む
from dotenv import load_dotenv
load_dotenv()

client = openai.OpenAI()

from utils import random_id, bootstrap_confidence_interval, get_examples, extract_total_score

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""以下のJSON形式で正確に回答してください。\n{str(request_keys)}\nリクエストフィールドを1つも欠かさず、適切に形成されたJSONオブジェクトであることを確認してください！\n"""
ROLE_DESC = lambda role: f"あなたは{role}です。"
SYSTEM_MSG = ""

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(
        msg,
        model,
        system_message,
        temperature=0.5
):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature, max_tokens=4096, stop=None, response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    # cost = response.usage.completion_tokens / 1000000 * 15 + response.usage.prompt_tokens / 1000000 * 5
    assert not json_dict is None
    return json_dict


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt_reflect(
        msg_list,
        model,
        temperature=0.8
):
    response = client.chat.completions.create(
        model=model,
        messages=msg_list,
        temperature=temperature, max_tokens=4096, stop=None, response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    assert not json_dict is None
    return json_dict


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_score_response_from_gpt_evaluation(
        msg,
        model,
        temperature=0.8
):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "あなたは役立つ助手です。"},
            {"role": "user", "content": msg},
        ],
        temperature=temperature, max_tokens=4096, stop=None
    )
    content = response.choices[0].message.content
    score = extract_total_score(content)
    assert not score is None
    return score

class LLMAgentBase():
    """
    LLM-Based Agentの基本クラス
    """

    def __init__(self, output_fields: list, agent_name: str,
                 role='helpful assistant', model='gpt-4o-mini-2024-07-18', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name

        self.role = role
        self.model = model
        self.temperature = temperature

        # give each instance a unique id
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction) -> str:
        # プロンプトの生成
        # システムプロンプトと入力情報を組み合わせて、LLMへの指示を作成
        output_fields_and_description = {key: f"Your {key}." for key in self.output_fields}
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description)

        # 入力情報テキストを構築する
        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + instruction
        return system_prompt, prompt

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> dict:
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        try:
            response_json = {}
            response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)
            assert len(response_json) == len(self.output_fields), "not returning enough fields"
        except Exception as e:
            # print(e)
            if "maximum context length" in str(e) and SEARCHING_MODE:
                raise AssertionError("The context is too long. Please try to design the agent to have shorter context.")
            # try to fill in the missing field
            for key in self.output_fields:
                if not key in response_json and len(response_json) < len(self.output_fields):
                    response_json[key] = ''
            for key in copy.deepcopy(list(response_json.keys())):
                if len(response_json) > len(self.output_fields) and not key in self.output_fields:
                    del response_json[key]
        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)


class AgentSystem():
    # エージェントシステムの基本クラス
    # 実際のforward関数はdynamically定義される
    def __init__(self) -> None:
        pass


def search(args):
    # メインの探索ループ
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    
    # 既存のアーカイブがあれば読み込み、なければ初期化
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        if "generation" in archive[-1] and isinstance(archive[-1]['generation'], int):
            start = archive[-1]['generation']
        else:
            start = 0
    else:
        archive = get_init_archive()
        start = 0

    # 初期アーカイブの評価
    for solution in archive:
        if 'fitness' in solution:
            continue

        solution['generation'] = "initial"
        print(f"============Initial Archive: {solution['name']}=================")
        try:
            acc_list = evaluate_forward_fn(args, solution["code"])
        except Exception as e:
            print("During evaluating initial archive:")
            print(e)
            continue

         # アーキテクチャの定性的な評価
        evaluation_prompt = get_evaluation_prompt(solution["thought"], solution["code"])
        score = get_score_response_from_gpt_evaluation(evaluation_prompt, args.model)

        # fitness_str = bootstrap_confidence_interval(acc_list)
        fitness_str = f"score: {score} / 80"
        print(fitness_str)
        solution['fitness'] = fitness_str

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(archive, json_file, indent=4, ensure_ascii=False)

    # メインの探索ループ
    for n in range(start, args.n_generation):
        print(f"============Generation {n + 1}=================")
        
        # プロンプトの生成と新しい解の取得
        system_prompt, prompt = get_prompt(archive)
        msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            # LLMを使用して新しい解を生成
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)

            # リフレクション（振り返り）プロセス
            Reflexion_prompt_1, Reflexion_prompt_2 = get_reflexion_prompt(archive[-1] if n > 0 else None)
            # リフレクション1
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_1})
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
            # リフレクション2
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_2})
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
        except Exception as e:
            print("新しい解の生成中にエラーが発生:")
            print(e)
            n -= 1
            continue

        # 新しい解の評価
        acc_list = []
        for _ in range(args.debug_max):
            try:
                acc_list = evaluate_forward_fn(args, next_solution["code"])
                if np.mean(acc_list) < 0.01 and SEARCHING_MODE:
                    raise Exception("全て0の精度")
                break
            except Exception as e:
                print("評価中にエラーが発生:")
                print(e)
                # エラーが発生した場合、デバッグのためのプロンプトを追加
                msg_list.append({"role": "assistant", "content": str(next_solution)})
                msg_list.append({"role": "user", "content": f"評価中にエラーが発生:\n{e}\n最新の実装で間違った箇所を慎重に検討してください。以前の試行からの洞察を使用して、現在のコードをデバッグし、同じ考えを実装してください。'thought'に以前の考えを繰り返し、'debug_thought'にデバッグのための考えを記入してください。"})
                try:
                    next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
                except Exception as e:
                    print("新しい解の生成中にエラーが発生:")
                    print(e)
                    continue
                continue
        if not acc_list:
            n -= 1
            continue

        # アーキテクチャの定性的な評価
        evaluation_prompt = get_evaluation_prompt(next_solution["thought"], next_solution["code"])
        score = get_score_response_from_gpt_evaluation(evaluation_prompt, args.model)

        # 新しい解のフィットネス計算と保存
        # fitness_str = bootstrap_confidence_interval(acc_list)
        fitness_str = f"score: {score} / 80"
        print(fitness_str)
        next_solution['fitness'] = fitness_str
        next_solution['generation'] = n + 1

        # 不要なフィールドの削除
        if 'debug_thought' in next_solution:
            del next_solution['debug_thought']
        if 'reflection' in next_solution:
            del next_solution['reflection']
        archive.append(next_solution)

        # 結果の保存
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(archive, json_file, indent=4, ensure_ascii=False)


# def evaluate(args):
#     file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
#     eval_file_path = str(os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")).strip(".json") + "_evaluate.json"
#     with open(file_path, 'r') as json_file:
#         archive = json.load(json_file)
#     eval_archive = []
#     if os.path.exists(eval_file_path):
#         with open(eval_file_path, 'r') as json_file:
#             eval_archive = json.load(json_file)

#     current_idx = 0
#     while (current_idx < len(archive)):
#         with open(file_path, 'r') as json_file:
#             archive = json.load(json_file)
#         if current_idx < len(eval_archive):
#             current_idx += 1
#             continue
#         sol = archive[current_idx]
#         print(f"current_gen: {sol['generation']}, current_idx: {current_idx}")
#         current_idx += 1
#         try:
#             acc_list = evaluate_forward_fn(args, sol["code"])
#         except Exception as e:
#             print(e)
#             continue
#         fitness_str = bootstrap_confidence_interval(acc_list)
#         sol['test_fitness'] = fitness_str
#         eval_archive.append(sol)

#         # save results
#         os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
#         with open(eval_file_path, 'w') as json_file:
#             json.dump(eval_archive, json_file, indent=4)


def evaluate_forward_fn(args, forward_str):
    # 動的にforward()関数を定義
    # https://github.com/luchris429/DiscoPOP/blob/main/scripts/launch_evo.py から修正
    namespace = {}
    exec(forward_str, globals(), namespace)  # 文字列からコードを実行し、新しい名前空間に格納
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)}個の要素がnamespaceにあります。1つだけ提供してください。")
    func = namespace[names[0]]  # 新しく定義された関数を取得
    if not callable(func):
        raise AssertionError(f"{func}は呼び出し可能ではありません")
    setattr(AgentSystem, "forward", func)  # AgentSystemクラスに動的にforwardメソッドを追加
    
    # set seed 0 for valid set
    # データセットの読み込みと準備
    examples = get_examples(args.data_filename)
    random.seed(args.shuffle_seed)
    random.shuffle(examples)

    # 検索モードか評価モードかで使用するデータを選択
    if SEARCHING_MODE:
        examples = examples[:args.valid_size] * args.n_repreat
    else:
        examples = examples[args.valid_size:args.valid_size + args.test_size] * args.n_repreat

    # 質問と回答の準備
    questions = [example['inputs'] for example in examples]
    answers = [example['targets'] for example in examples]

    print(f"問題数: {len(examples)}")
    
    # マルチプロセッシングの設定
    max_workers = min(len(examples), args.max_workers) if args.multiprocessing else 1

    # タスクキューの準備
    task_queue = []
    for q in questions:
        taskInfo = Info('task', 'User', q, -1)
        task_queue.append(taskInfo)

    agentSystem = AgentSystem()

    # 精度計算用のリスト
    acc_list = []
    
    # マルチスレッドで予測を実行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(agentSystem.forward, task_queue), total=len(task_queue)))

    # 結果の処理と精度の計算
    for q_idx, res in enumerate(results):
        # resが取得できていれば成功とする
        try:
            if isinstance(res, Info):
                extracted_answer = res.content
            else:
                extracted_answer = res
        except Exception as e:
            acc_list.append(0)
            continue

        acc_list.append(1)
    
    # 最終的な精度を表示
    print(f"精度: {bootstrap_confidence_interval(acc_list)}")
    return acc_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_filename', type=str, default="dataset/coding_test.jsonl")
    parser.add_argument('--valid_size', type=int, default=1)
    parser.add_argument('--test_size', type=int, default=0)
    parser.add_argument('--shuffle_seed', type=int, default=0)
    parser.add_argument('--n_repreat', type=int, default=1)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=48)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--expr_name', type=str, default="coding_gpt4omini_results")
    parser.add_argument('--n_generation', type=int, default=30)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--model',
                        type=str,
                        default='gpt-4o-mini-2024-07-18',
                        choices=['gpt-4o-mini-2024-07-18', 'gpt-4o-2024-08-06'])

    args = parser.parse_args()
    # search
    SEARCHING_MODE = True
    search(args)

    # evaluate
    # SEARCHING_MODE = False
    # evaluate(args)
