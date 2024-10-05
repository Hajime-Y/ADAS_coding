import json

EXAMPLE = {
    "thought": "**洞察:**\n次の興味深いエージェントについてのあなたの洞察。\n**全体的なアイデア:**\nエージェント設計の背後にある理由付けと全体的なコンセプト。\n**実装:**\n実装を段階的に説明してください。",
    "name": "提案するエージェントの名前",
    "code": """def forward(self, taskInfo):
    # ここにコードを記述
    return answer
"""
}

COT = {
    "thought": "連鎖的思考（Chain-of-Thought, CoT）によって、LLMが直接答えを出力するのではなく、考える過程を一歩一歩進めることで、複雑な問題解決を可能にします。この手法により、モデルはより深い理解を必要とするタスクに対応し、その決定過程を理解することができます。",
    "name": "Chain-of-Thought",
    "code": """def forward(self, taskInfo):
    # Chain-of-Thought (CoT) アプローチのための指示
    # これは、LLMがタスクを解く前に考える過程を持つことを可能にする重要な手法です。
    cot_instruction = "ステップバイステップで考え、タスクを解いてください。"

    # CoT 専用の新しい LLM エージェントをインスタンス化
    # LLM が答える前に考える過程を持たせるには、追加の出力フィールド 'thinking' を設定する必要があります。
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')

    # CoT エージェントの入力を準備
    # 入力は Info のリストであり、最初の要素は通常 taskInfo です
    cot_agent_inputs = [taskInfo]

    # CoT エージェントからの応答を取得
    thinking, answer = cot_agent(cot_agent_inputs, cot_instruction)

    # 最終的な答えのみを返す
    return answer
"""
}

COT_SC = {"thought": "LLMは正しい答えに到達することができますが、その理由付けは異なる場合があります。高温設定で同じ質問を繰り返し尋ねることで、異なる理由付けのパスを生成します。そして、複数の Chain-of-Thought (CoT) エージェントから得られた複数の答えを組み合わせて、アンサンブルによってより正確な最終的な答えを得ます。",
          "name": "Self-Consistency with Chain-of-Thought",
          "code": """def forward(self, taskInfo):
    # ステップバイステップの推論のための指示
    cot_instruction = "ステップバイステップで考えてからタスクを解いてください。"
    N = 5 # CoT エージェントの数

    # 異なる理由付けのために高温設定で複数の CoT エージェントを初期化
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', temperature=0.8) for _ in range(N)]

    # 収集された推論と回答に基づく最終決定のための指示
    final_decision_instruction = "上記のすべての解決策を考慮し、慎重に推論して最終的な答えを提供してください。"
    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', temperature=0.1)
    
    possible_answers = []
    for i in range(N):
        thinking, answer = cot_agents[i]([taskInfo], cot_instruction)
        possible_answers.extend([thinking, answer])

    # 生成されたすべての回答に基づいて最終決定を行う
    thinking, answer = final_decision_agent([taskInfo] + possible_answers, final_decision_instruction)
    return answer  
"""
}

Reflexion = {
    "thought": "パフォーマンスを向上させるため、LLMはフィードバックに基づいて反復的に答えを改善できます。前回の試行とフィードバックを反映させ、モデルはその理解を改善し、より正確な解決策を提供できます。",
    "name": "Self-Refine (Reflexion)",
    "code": """def forward(self, taskInfo):
    # 初期の理解のための指示
    cot_initial_instruction = "ステップバイステップで考えてからタスクを解いてください。"

    # 前回の試行とフィードバックに基づいて改善するための指示
    cot_reflect_instruction = "前回の試行とフィードバックを考慮し、最新の試行で間違える可能性がある箇所を慎重に検討してください。前回の試行から得られた洞察を活用し、タスクをより良く解決してください。"
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')

    # 答えをフィードバックし、修正するための指示
    critic_instruction = "上記の答えを再度見直し、間違っている可能性がある箇所を批判してください。絶対に正しいと確信できる場合は、'correct' に 'True' を出力してください。"
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent')
    
    N_max = 5 # 最大試行回数

    # 初回の試行
    cot_inputs = [taskInfo]
    thinking, answer = cot_agent(cot_inputs, cot_initial_instruction, 0)

    for i in range(N_max):
        # 批判者からフィードバックと正解ステータスを取得
        feedback, correct = critic_agent([taskInfo, thinking, answer], critic_instruction, i)
        if correct.content == 'True':
            break
            
        # 次回の試行の入力にフィードバックを追加
        cot_inputs.extend([thinking, answer, feedback])

        # 前回の試行を反映して答えを改善
        thinking, answer = cot_agent(cot_inputs, cot_reflect_instruction, i + 1)
    return answer
"""
}

LLM_debate = {
    "thought": "異なる LLM が互いに議論することで、彼らの様々な視点を活用してタスクに対するより良い解決策を見つけることができます。",
    "name": "LLM Debate",
    "code": """def forward(self, taskInfo):
    # 初期の理解のための指示
    debate_initial_instruction = "ステップバイステップで考えてからタスクを解いてください。"

    # 他のエージェントの解決策に基づいて議論し、解決策を更新するための指示
    debate_instruction = "他のエージェントからの問題に対する解決策を考慮し、その意見を追加のアドバイスとして慎重に検討してください。更新された答えを提供してください。"
    
    # 異なる役割と中程度の温度設定で様々な視点を持つ議論エージェントを初期化
    debate_agents = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', temperature=0.8, role=role) for role in ['Biology Expert', 'Physics Expert', 'Chemistry Expert', 'Science Generalist']]

    # 全議論結果と解決策に基づいて最終的な決定を下すための指示
    final_decision_instruction = "全ての思考と答えを慎重に検討し、最終的な答えを提供してください。"
    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', temperature=0.1)

    max_round = 2 # 最大議論ラウンド数
    all_thinking = [[] for _ in range(max_round)]
    all_answer = [[] for _ in range(max_round)]

    # 議論ラウンドを実施
    for r in range(max_round):
        for i in range(len(debate_agents)):
            if r == 0:
                thinking, answer = debate_agents[i]([taskInfo], debate_initial_instruction)
            else:
                input_infos = [taskInfo] + [all_thinking[r-1][i]] + all_thinking[r-1][:i] + all_thinking[r-1][i+1:]
                thinking, answer = debate_agents[i](input_infos, debate_instruction)
            all_thinking[r].append(thinking)
            all_answer[r].append(answer)
    
    # 全議論結果と解決策に基づいて最終的な決定を下す
    thinking, answer = final_decision_agent([taskInfo] + all_thinking[max_round-1] + all_answer[max_round-1], final_decision_instruction)
    return answer
"""
}

Take_a_step_back = {"thought": "LLMがタスクを解く上で役立つ原理を最初に理解するようにしましょう。タスクに関連する原理を理解することで、モデルは問題をより深く理解し、より正確な解決策を提供できます。",
                    "name": "Step-back Abstraction",
                    "code": """def forward(self, taskInfo):
        # タスクに関連する原理を理解するための指示
        principle_instruction = "このタスクを解決するために必要な、システムアーキテクチャ、コーディング、UX/UI設計、AI/ML工学の観点から重要な概念や原理を考えてください。まずはステップバイステップで考えてから、各分野に関連する全ての重要な概念を列挙して説明してください。"
        
        # 原理に基づいてタスクを解くための指示
        cot_instruction = "問題とその背後にある原理を考えてから、ステップバイステップで考えてタスクを解いてください。"
        
        # LLM エージェントをインスタンス化
        principle_agent = LLMAgentBase(['thinking', 'principle'], 'Principle Agent')
        cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')
        
        # タスクに関連する原理を取得
        thinking, principle = principle_agent([taskInfo], principle_instruction)

        # 原理を用いてタスクを解く
        thinking, answer = cot_agent([taskInfo, thinking, principle], cot_instruction)
        return answer
"""
                    }

QD = {"thought": "Quality-Diversity メソッドと同様に、LLMが複数の多様な解決策を生成することで役立つ場合があります。モデルに異なる理由付けのパスを探索させることで、最適な解決策を見つける可能性が増えます。",
      "name": "Quality-Diversity",
      "code": """def forward(self, taskInfo):
    # 初期の理解のための指示
    cot_initial_instruction = "考える過程を一歩一歩進めてからタスクを解いてください。"

    # 多様な答えを生成するための指示
    qd_instruction = "前回の試行を考慮し、タスクを解く別の興味深い方法を考えてください。"
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')

    # 収集された理由付けと答えに基づいて最終的な決定を下すための指示
    final_decision_instruction = "全ての解決策を慎重に検討し、最終的な答えを提供してください。"
    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', temperature=0.1)
    
    N_max = 3 # 最大試行回数

    # 初回の試行
    cot_inputs = [taskInfo]
    possible_answers = []
    thinking, answer = cot_agent(cot_inputs, cot_initial_instruction, 0)

    # 答えを可能性のある答えのリストに追加
    possible_answers.extend([thinking, answer])

    for i in range(N_max):
        # 前回の試行を反映し、別の興味深い答えを生成
        cot_inputs.extend([thinking, answer])

        # 別の興味深い答えを生成
        thinking, answer = cot_agent(cot_inputs, qd_instruction, i + 1)
        possible_answers.extend([thinking, answer])

    # 全ての生成された答えに基づいて最終的な決定を下す
    thinking, answer = final_decision_agent([taskInfo] + possible_answers, final_decision_instruction)
    return answer
"""
      }

Role_Assignment = {"thought": "Auto-GPT や専門家のプロンプトと同様に、システムの設計に動的な制御フローを使用して、どの専門家を使用すべきかをエージェントに決定させることができます。",
                   "name": "Dynamic Assignment of Roles",
                   "code": """def forward(self, taskInfo):
        # ステップバイステップの理解のための指示
        cot_instruction = "ステップバイステップで考えてからタスクを解いてください。"
        expert_agents = [LLMAgentBase(['thinking', 'answer'], 'Expert Agent', role=role) for role in ['System Architect', 'Coding Expert', 'UX/UI Designer', 'AI/ML Engineer', 'Full-Stack Engineer']]

        # タスクを適切な専門家にルーティングするための指示
        routing_instruction = "タスクを考慮し、問題に答える専門家を選んでください。System Architect Expert、Coding Expert、UX/UI Design Expert、または AI and Machine Learning Expert から選択してください。"
        routing_agent = LLMAgentBase(['choice'], 'Routing agent')

        # タスクをルーティングする専門家の選択を取得
        choice = routing_agent([taskInfo], routing_instruction)[0]

        if 'architect' in choice.content.lower():
            expert_id = 0
        elif 'coding' in choice.content.lower():
            expert_id = 1
        elif 'ux/ui' in choice.content.lower():
            expert_id = 2
        elif 'ai/ml' in choice.content.lower():
            expert_id = 3
        else:
            expert_id = 4 # デフォルトで Full-Stack Engineer

        thinking, answer = expert_agents[expert_id]([taskInfo], cot_instruction)
        return answer
"""
                   }

system_prompt = """あなたは役立つ助手です。必ず整った形式の JSON オブジェクトとして返答してください。"""

task_analysis_prompt = """# 概要:
あなたは機械学習の研究者で、様々なエージェント系をテストしています。あなたの目的は、これらのシステム内でブロックを設計することです。あなたの目標は、『複数ファイルを必要とする複雑なソフトウェアを構築』できるで優れたエージェントを設計することです。これは、プログラミングのみならずソフトウェア開発全体の幅広い能力を必要とする厳しいタスクです。
あなたはまず、過去のアーキテクチャアーカイブを元に、この『複数ファイルを必要とする複雑なソフトウェアを構築』をマルチエージェントシステムが実行するのに有力なタスクのステップを考えてください。

## タスク例:
- Webスクレイピングと自然言語処理を組み合わせた株価分析システムを作成せよ。複数の金融ニュースサイトから情報を収集し、センチメント分析を行い、その結果を基に株価予測を行うこと。
- マイクロサービスアーキテクチャを用いた、スケーラブルなeコマースプラットフォームを設計・実装せよ。ユーザー認証、商品管理、注文処理、在庫管理、決済システムを個別のマイクロサービスとして実装すること。
- マルチプラットフォーム対応の暗号通貨ウォレットアプリケーションを開発せよ。Web、iOS、Android向けのクライアントアプリケーションと、セキュアなバックエンドサーバー、ブロックチェーンとの連携機能を実装すること。

# 発見されたアーキテクチャアーカイブ:
ここに発見されたアーキテクチャのアーカイブがあります:

[ARCHIVE]

適合度の値は、構築されたソフトウェアを機能性、信頼性、ユーザービリティ、保守性の観点で評価された値です。あなたの目標は "適合度" を最大化することです。

# 出力指示と例:
最初のキーは ("thought") で、この『複数ファイルを必要とする複雑なソフトウェアを構築』を実行するのに有力なタスクのステップを決めるためのあなたの思考過程を捕捉する必要があります。"thought" セクションでは、アーカイブを確認して有力なタスクを学び、それを踏まえた上で次に試すべきより興味深いタスクを考え、その理由付けと全体的なコンセプトを説明します。
2番目のキー ("tasks") は、タスクのステップを詳細に書かれている必要があります。

# あなたのタスク:
あなたは 『複数ファイルを必要とする複雑なソフトウェアを構築』作業を深く理解しています。あなたの目標は、興味深く新しいより複雑なエージェントを提案することで "適合度" を最大化することです。
そのために、まず『複数ファイルを必要とする複雑なソフトウェアを構築』タスクを分割し、 "適合度" の最大化につながるタスクの分割を発見していきます。
アーカイブ内のアーキテクチャを注意深く観察し、そこからどのような洞察、教訓、あるいは今後の発展につながる重要な要素が得られるかについて考えてください。
興味深い次のタスク分割を考えるために創造的に考えてください。
既成の枠や既成概念に囚われずに考えてください。
"""

expert_identification_prompt = """# 概要:
あなたは機械学習の研究者で、様々なエージェント系をテストしています。あなたの目的は、これらのシステム内でブロックを設計することです。あなたの目標は、『複数ファイルを必要とする複雑なソフトウェアを構築』できるで優れたエージェントを設計することです。これは、プログラミングのみならずソフトウェア開発全体の幅広い能力を必要とする厳しいタスクです。
あなたはまず、過去のアーキテクチャアーカイブを元に、この『複数ファイルを必要とする複雑なソフトウェアを構築』をマルチエージェントシステムが実行するのに必要な専門家をリストアップしてください。

## タスク例:
- Webスクレイピングと自然言語処理を組み合わせた株価分析システムを作成せよ。複数の金融ニュースサイトから情報を収集し、センチメント分析を行い、その結果を基に株価予測を行うこと。
- マイクロサービスアーキテクチャを用いた、スケーラブルなeコマースプラットフォームを設計・実装せよ。ユーザー認証、商品管理、注文処理、在庫管理、決済システムを個別のマイクロサービスとして実装すること。
- マルチプラットフォーム対応の暗号通貨ウォレットアプリケーションを開発せよ。Web、iOS、Android向けのクライアントアプリケーションと、セキュアなバックエンドサーバー、ブロックチェーンとの連携機能を実装すること。

# 新しいアーキテクチャで想定しているタスク分割:
新しく作っていくアーキテクチャは以下のタスク分割でエージェントが構築される予定です。

[TASKS]

これを踏まえて必要な専門家を考えます。

# 発見されたアーキテクチャアーカイブ:
ここに発見されたアーキテクチャのアーカイブがあります:

[ARCHIVE]

適合度の値は、構築されたソフトウェアを機能性、信頼性、ユーザービリティ、保守性の観点で評価された値です。あなたの目標は "適合度" を最大化することです。

# 出力指示と例:
最初のキーは ("thought") で、この『複数ファイルを必要とする複雑なソフトウェアを構築』を実行するのに有力なタスクのステップを決めるためのあなたの思考過程を捕捉する必要があります。"thought" セクションでは、アーカイブを確認して有力なタスクを学び、それを踏まえた上で次に試すべきより興味深いタスクを考え、その理由付けと全体的なコンセプトを説明します。
2番目のキー ("experts") は、必要な専門家エージェントについてリストアップされており、詳細に書かれている必要があります。

# あなたのタスク:
あなたは 『複数ファイルを必要とする複雑なソフトウェアを構築』作業を深く理解しています。あなたの目標は、興味深く新しいより複雑なエージェントを提案することで "適合度" を最大化することです。
そのために、まず『複数ファイルを必要とする複雑なソフトウェアを構築』に必要な専門家を特定・リストアップし、 "適合度" の最大化につながる専門家エージェントを発見していきます。
アーカイブ内のアーキテクチャを注意深く観察し、そこからどのような洞察、教訓、あるいは今後の発展につながる重要な要素が得られるかについて考えてください。
興味深い次の専門家エージェントを考えるために創造的に考えてください。
既成の枠や既成概念に囚われずに考えてください。
"""

base = """# 概要
あなたは機械学習の研究者で、様々なエージェント系をテストしています。あなたの目的は、これらのシステム内でブロックを設計することです。あなたの目標は、複数ファイルを必要とする複雑なソフトウェアを構築できるで優れたエージェントを設計することです。これは、プログラミングのみならずソフトウェア開発全体の幅広い能力を必要とする厳しいタスクです。

## タスク例:

- Webスクレイピングと自然言語処理を組み合わせた株価分析システムを作成せよ。複数の金融ニュースサイトから情報を収集し、センチメント分析を行い、その結果を基に株価予測を行うこと。
- マイクロサービスアーキテクチャを用いた、スケーラブルなeコマースプラットフォームを設計・実装せよ。ユーザー認証、商品管理、注文処理、在庫管理、決済システムを個別のマイクロサービスとして実装すること。
- マルチプラットフォーム対応の暗号通貨ウォレットアプリケーションを開発せよ。Web、iOS、Android向けのクライアントアプリケーションと、セキュアなバックエンドサーバー、ブロックチェーンとの連携機能を実装すること。

# ユーティリティコード:

```python
from collections import namedtuple
from typing import Union
import numpy as np
import json

import openai
import backoff
from utils import random_id

# OpenAI クライアントを初期化
client = openai.OpenAI()

# タスク情報を保持するための名前付きタプル
Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

# LLM 応答のフォーマットを整形
FORMAT_INST = lambda request_keys: f"以下の JSON フォーマットで必ず正確に返答してください。\n{str(request_keys)}\nフィールドを欠落させず、JSON フォーマットが正しいことを確認してください！\n"

# LLM の役割を記述
ROLE_DESC = lambda role: f"あなたは {role} です。"

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(msg, model, system_message, temperature=0.5):
    \"""
    GPT モデルから JSON 応答を取得する関数。
    
    Args:
    - msg (str): ユーザーメッセージ。
    - model (str): 使用するモデル。
    - system_message (str): システムメッセージ。
    - temperature (float): サンプリング温度。
    
    Returns:
    - dict: JSON 応答。
    \"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature,
        max_tokens=1024,
        stop=None,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    return json_dict

class LLMAgentBase:
    \"""
    LLM エージェントのベースクラス。
    
    Attributes:
    - output_fields (list): 出力に期待されるフィールド。
    - agent_name (str): エージェントの名前。
    - role (str): エージェントの役割記述。
    - model (str): 使用するモデル。(オプション。デフォルトのままにしてください。)
    - temperature (float): サンプリング温度。
    - id (str): エージェントインスタンスの一意の識別子。
    \"""

    def __init__(self, output_fields: list, agent_name: str, role='helpful assistant', model='gpt-4o-mini-2024-07-18', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.id = random_id()
    
    def generate_prompt(self, input_infos, instruction) -> str:
        \"""
        LLM のためのプロンプトを生成します。
        
        Args:
        - input_infos (list): 入力情報のリスト。
        - instruction (str): タスクの指示。
        
        Returns:
        - tuple: システムプロンプトとユーザープロンプト。

        生成されるプロンプトの例:
        ""
        あなたは役立つアシスタントです。
        
        # 出力フォーマット:
        以下の JSON フォーマットで必ず正確に返答してください。
        ...

        # あなたのタスク:
        あなたはいくつかのペアの入力と出力を与えられます。出力 ...

        ### thinking #1 by Chain-of-Thought Agent hkFo (yourself):
        ...
        
        ### code #1 by Chain-of-Thought Agent hkFo (yourself):
        ...

        ### answer by Chain-of-Thought Agent hkFo's code evaluator:...


        # 指示: 
        ステップバイステップで考えてから、コードを書いてタスクを解いてください。
        ""
        \"""
        output_fields_and_description = {key: f"あなたの {key}." for key in self.output_fields}
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description)

        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# あなたのタスク:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx+1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + instruction
        return system_prompt, prompt 

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> list[Info]:
        \"""
        LLM に提供された入力情報と指示でクエリを実行します。
        
        Args:
        - input_infos (list): 入力情報のリスト。
        - instruction (str): タスクの指示。
        - iteration_idx (int): タスクのイテレーションインデックス。
        
        Returns:
        - output_infos (list[Info]): 出力情報。
        \"""
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)

        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"
    
    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        # 注意:
        # LLM の出力は Info のリストです。単一の出力をクエリする場合は [0] でアクセスしてください。
        # 'thinking' を出力に含めることが一般的な良い習慣です。
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)

class AgentArchitecture:
    \"""
    ここにコードを埋め込んでください。
    \"""
    def forward(self, taskInfo) -> Union[Info, str]:
        \"""
        タスク情報を処理するためのプレースホルダーメソッド。
        
        Args:
        - taskInfo (Info): タスク情報。
        
        Returns:
        - Answer (Union[Info, str]): あなたの最終的な答え。Info の名前付きタプルまたは文字列の答えを返してください。
        \"""
        pass
```
# 発見されたアーキテクチャアーカイブ
ここに発見されたアーキテクチャのアーカイブがあります:

[ARCHIVE]

適合度の値は、構築されたソフトウェアを機能性、信頼性、ユーザービリティ、保守性の観点で評価された値です。あなたの目標は "適合度" を最大化することです。

# 出力指示と例:
最初のキーは ("thought") で、次の機能を設計するためのあなたの思考過程を捕捉する必要があります。"thought" セクションでは、次に試すべき興味深いエージェントを考え、その理由付けと全体的なコンセプトを説明し、最後に実装手順を詳細に説明します。
2番目のキー ("name") は、次のエージェントアーキテクチャの名前に対応します。
最後に、3番目のキー ("code") は、実際の Python コードの "forward()" 関数に対応します。"code" には、プロジェクト全体の一部となる完全なコードを書く必要があります。

ここに次のエージェントアーキテクチャの出力フォーマットの例があります:

[EXAMPLE]

上記の関数インターフェースを使用する必要があります。各 LLM エージェントに、それぞれの役割を果たすために必要な指示、入力情報、必要な出力フィールドを指定する必要があります。
また、LLM の役割と温度をさらに制御することで、LLM の応答をさらに制御できます。LLMAgentBase() は自動的に出力を解析し、Info のリストを返します。Info.content でコンテンツを取得できます。
タスクについて LLM に知らせる必要があると思われる場合は、タスク情報を含めることを忘れないでください。

## 間違った実装例:
これらが間違った実装の例です:

1. これは間違っています: ```
feedback, correct = critic_agent([taskInfo, thinking, answer], critic_instruction, i)
feedback_info = verifier_agent([taskInfo, Info('feedback', 'Critic Agent', thinking, 0)], verification_instruction)
```
"Info('feedback', 'Critic Agent', thinking, 0)" を使用するのは間違っています。LLMAgentBase から返される "feedback" はすでに Info です。

2. これは間違っています: ```
# Debugging: Log the generated answer
print('Generated Answer:', ...)
feedback_info = verifier_agent([taskInfo, Info('feedback', 'Critic Agent', thinking, 0)], verification_instruction)
if len(feedback_info) < 3:  # Check if feedback_info has enough elements
    return 'Error: Feedback info incomplete'
```
まず、len(feedback_info) は機能しません。
次に、エラーメッセージを返すべきではありません。常に最善の答えを返す必要があります。
3番目に、コード内に何も print してはいけません。
最後に、再度 Info オブジェクトを自分で作成しないでください。

3. これは間違っています: ```
all_thinking = []
all_answers = []
for agent, role in zip(agents, roles):
    outputs = agent([taskInfo], independent_reasoning_instruction.format(role=role))
    all_thinking.append(outputs[0].content)
    all_answers.append(outputs[1].content)

# Aggregate the reasoning paths and answers
aggregated_thinking = '\n'.join(all_thinking)
aggregated_answers = '\n'.join(all_answers)
```
Info オブジェクトからコンテンツを自分で抽出してはいけません。Info オブジェクトをそのまま使用してください。コンテンツを集約する場合は、単に Info オブジェクトをリストに入れ、そのリストを次の LLM エージェントの入力として使用してください。

4. これは間違っています: ```
reasoning_agent = LLMAgentBase(['thinking', 'answer'], 'Reasoning Agent')
response_infos = reasoning_agent([taskInfo] + ..., reasoning_instruction)
    
# Extract the final answer from the response_infos
for info in response_infos:
    if info.name == 'final_answer':
        return info
# Fallback if no answer is found
return Info('answer', 'Final Decision Agent', 'No answer generated.', 0)
```
最終的な答えを自分で抽出してはいけません。最終的な答えの Info を直接返す必要があります。また、常に最善の答えを返す必要があります。
正しい例: ```
reasoning_agent = LLMAgentBase(['thinking', 'answer'], 'Reasoning Agent')
thinking, answer = reasoning_agent([taskInfo] + ..., reasoning_instruction)
return answer
```

# あなたのタスク
あなたは LLM プロンプト技術と LLM エージェントの仕組みについて文献から深く理解しています。あなたの目標は、興味深く新しいより複雑なエージェントを提案することで "適合度" を最大化することです。
アーカイブ内のアーキテクチャを注意深く観察し、そこからどのような洞察、教訓、あるいは今後の発展につながる重要な要素が得られるかについて考えてください。
興味深い次のアーキテクチャを考えるために創造的に考えてください。関連する LLM エージェントの論文や他の研究分野の学術論文からインスピレーションを得ることをお勧めします。
学んだ知識と学術文献のインスピレーションを活用して、次の興味深いアーキテクチャを提案してください。
既成の枠や既成概念に囚われずに考えてください。
"""

Reflexion_prompt_1 = f""""[EXAMPLE]提案された新しいアーキテクチャを注意深く見直して、以下の点について考えてください:"

1. **興味深さ**: 提案されたアーキテクチャがアーカイブ内の既存の方法と比較して興味深いか革新的かどうかを評価してください。提案されたアーキテクチャが興味深くないと判断した場合は、これらの欠点を解消する新しいアーキテクチャを提案してください。 
- 提案されたアーキテクチャと前回の試行との違いを確認してください。
- 提案とアーカイブ内のアーキテクチャを注意深く比較してください。実装における実際の違いを含めてください。
- 現在のアーキテクチャが革新的かどうかを決定してください。
- 批判的思考を使用してください！

2. **実装の間違い**: 実装における間違いを特定し、コードを注意深く見直し、発見した問題をデバッグし、修正版を提供してください。"## 間違った実装例" のセクションをプロンプトで確認してください。

3. **改善**: 提案されたアーキテクチャに基づいて、実装を改善してパフォーマンスや効果を向上させることができる改善点を提案してください。このステップでは、実装の詳細を細かく調整してアーキテクチャの全体的な設計フレームワークを変更することは避けてください。ただし、現在のアーキテクチャが興味深くない場合は、新しいアーキテクチャを提案することができます。
- 実装が実際に行っていることを注意深く観察してください。
- 実装に冗長なコードや不必要なステップがある場合は、効果的な実装に置き換えてください。
- 実装が前のエージェントと似すぎないようにしてください。

そして、反映された改善版または新しいアーキテクチャの実装を提供する必要があります。

あなたの応答は以下の形式で構成される必要があります:

"reflection": アーキテクチャの興味深さについての考え、実装の間違いを特定し、改善点を提案してください。

"thought": 前回の提案を改善するか、必要に応じて新しいアーキテクチャを提案してください。例の応答と同じ形式を使用してください。

"name": 改善または新しいアーキテクチャの名前を提供してください。(名前に "new" や "improved" などの単語を含めないでください。)

"code": 修正されたコードまたは改善された実装を提供してください。実際に修正と改善をこのコードに実装してください。
"""

Reflexion_prompt_2 = """"## 間違った実装例" のセクションのヒントに従って、コードをさらに修正してください。
あなたの応答は以下の形式で構成される必要があります:
新しい反省的思考を "reflection" に入れてください。前回の "thought" と "name" を繰り返し、修正されたバージョンのコードを "code" に更新してください。
"""

Evaluation_prompt = """あなたは、LLMベースのマルチエージェントシステムの設計と実装を評価する専門家です。以下の情報が与えられます。

### 1. アーキテクチャの考え方：[THOUGHT]

### 2. アーキテクチャ：```[CODE]```

### 3. マルチエージェントシステムのタスク：複数ファイルを必要とする複雑なソフトウェアを構築することです。これは、プログラミングのみならずソフトウェア開発全体の幅広い能力を必要とする厳しいタスクです。

#### タスク例:
- Webスクレイピングと自然言語処理を組み合わせた株価分析システムを作成せよ。複数の金融ニュースサイトから情報を収集し、センチメント分析を行い、その結果を基に株価予測を行うこと。
- マイクロサービスアーキテクチャを用いた、スケーラブルなeコマースプラットフォームを設計・実装せよ。ユーザー認証、商品管理、注文処理、在庫管理、決済システムを個別のマイクロサービスとして実装すること。
- マルチプラットフォーム対応の暗号通貨ウォレットアプリケーションを開発せよ。Web、iOS、Android向けのクライアントアプリケーションと、セキュアなバックエンドサーバー、ブロックチェーンとの連携機能を実装すること。

これらの情報を基に、以下の評価基準に従ってシステムを評価してください。各項目を1-10のスケールで評価し、詳細なコメントを提供してください。

## 評価基準

1. アーキテクチャの適合性 (20点満点)
   a) タスクの性質と複雑さに対するアーキテクチャの適切さ (10点)
   b) アーキテクチャの新規性と独創性 (10点)

2. タスク分割の効果性 (10点満点)
   a) メインタスクを詳細なタスクに適切に分割できているか (10点)

3. エージェント設計の明確性 (30点満点)
   a) 各エージェントの役割定義の明確さ (10点)
   b) エージェント間の役割分担の適切さ (10点)
   c) 特定領域のExpertを用いている場合、メインタスクとExpertとの過不足・整合性 (10点)

4. プロンプト設計の質 (20点満点)
   a) 各エージェントのプロンプトと役割の整合性 (10点)
   b) プロンプトの指示の明確さと具体性 (10点)

評価スケール：
- 1-2: 欠陥がある
- 3-4: 標準的な水準
- 5-6: 優秀
- 7-8: 極めて優秀（改善点が見当たらない）
- 9-10: 人類の限界を超えている（10は人類が到達可能な限界点であり、極めて稀）

## 評価例

例えば、以下のアーキテクチャを評価すると全項目4点で、総合得点 32 / 80 となります。これと比較して優れているか否かを評価指標の１つとしてください。
### 1. アーキテクチャの考え方：LLMがタスクを解く上で役立つ原理を最初に理解するようにしましょう。タスクに関連する原理を理解することで、モデルは問題をより深く理解し、より正確な解決策を提供できます。
### 2. アーキテクチャ：```def forward(self, taskInfo):\n        # タスクに関連する原理を理解するための指示\n        principle_instruction = \"このタスクを解決するために必要な、システムアーキテクチャ、コーディング、UX/UI設計、AI/ML工学の観点から重要な概念や原理を考えてください。まずはステップバイステップで考えてから、各分野に関連する全ての重要な概念を列挙して説明してください。\"\n        \n        # 原理に基づいてタスクを解くための指示\n        cot_instruction = \"問題とその背後にある原理を考えてから、ステップバイステップで考えてタスクを解いてください。\"\n        \n        # LLM エージェントをインスタンス化\n        principle_agent = LLMAgentBase(['thinking', 'principle'], 'Principle Agent')\n        cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')\n        \n        # タスクに関連する原理を取得\n        thinking, principle = principle_agent([taskInfo], principle_instruction)\n\n        # 原理を用いてタスクを解く\n        thinking, answer = cot_agent([taskInfo, thinking, principle], cot_instruction)\n        return answer\n```

## 出力形式

全体の考察、確認、検討後、以下の形式で評価結果を出力してください：

```
評価結果:

1. アーキテクチャの適合性
   a) タスクの性質と複雑さに対するアーキテクチャの適切さ: [得点]/10
   b) アーキテクチャの新規性と独創性: [得点]/10

2. タスク分割の効果性
   a) メインタスクを詳細なタスクに適切に分割できているか: [得点]/10

3. エージェント設計の明確性
   a) 各エージェントの役割定義の明確さ: [得点]/10
   b) エージェント間の役割分担の適切さ: [得点]/10
   c) 特定領域のExpertを用いている場合、メインタスクとExpertとの過不足・整合性: [得点]/10

4. プロンプト設計の質
   a) 各エージェントのプロンプトと役割の整合性: [得点]/10
   b) プロンプトの指示の明確さと具体性: [得点]/10

総合得点: [総合得点]/80
```

注意事項：
- 評価は客観的かつ公平に行ってください。
- 10点は極めて稀で、革新的かつ完璧な実装にのみ与えられます。
- コードの実行可能性は既に確認されているため、コードの動作そのものではなく、設計と実装の質に焦点を当ててください。
- 出力形式を厳密に守り、各セクションを明確に区別してください。これにより、評価結果の自動抽出が容易になります。

この評価結果は、システムの改善と最適化に使用されます。建設的かつ詳細なフィードバックを提供してください。
"""


def get_init_archive():
    return [COT, COT_SC, Reflexion, LLM_debate, Take_a_step_back, QD, Role_Assignment]


def get_prompt(current_archive, adaptive=False):
    archive_str = ",\n".join([json.dumps(sol) for sol in current_archive])
    archive_str = f"[{archive_str}]"
    prompt = base.replace("[ARCHIVE]", archive_str)
    prompt = prompt.replace("[EXAMPLE]", json.dumps(EXAMPLE))

    return system_prompt, prompt


def get_reflexion_prompt(prev_example):
    prev_example_str = "これが前回試したエージェントです:\n" + json.dumps(prev_example) + "\n\n"
    r1 = Reflexion_prompt_1.replace("[EXAMPLE]", prev_example_str) if prev_example else Reflexion_prompt_1.replace("[EXAMPLE]", "")
    return r1, Reflexion_prompt_2


def get_evaluation_prompt(prev_thought, prev_code):
    prompt = Evaluation_prompt.replace("[THOUGHT]", prev_thought).replace("[CODE]", prev_code)
    return prompt

