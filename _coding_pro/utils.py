import random
import string
from collections import namedtuple

import numpy as np
import json
import re

# Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])

# QUERY_TEMPLATE_MULTICHOICE = """
# Answer the following multiple choice question.

# {Question}

# (A) {A}
# (B) {B}
# (C) {C}
# (D) {D}
# """.strip()


# def format_multichoice_question(row):
#     return QUERY_TEMPLATE_MULTICHOICE.format(**row)


def random_id(length=4):
    characters = string.ascii_letters + string.digits  # includes both upper/lower case letters and numbers
    random_id = ''.join(random.choices(characters, k=length))
    return random_id


def bootstrap_confidence_interval(data, num_bootstrap_samples=100000, confidence_level=0.95):
    """
    1次元の精度データの平均値に対するブートストラップ信頼区間を計算します。
    また、ブートストラップ平均値の中央値も返します。
    
    引数:
    - data (list or array of float): 1次元のデータポイントのリストまたは配列。
    - num_bootstrap_samples (int): ブートストラップサンプルの数。
    - confidence_level (float): 希望する信頼水準（例：95%の場合は0.95）。
    
    戻り値:
    - str: 95%信頼区間と中央値をパーセンテージで小数点以下1桁まで表示した書式化された文字列。
    """
    # Convert data to a numpy array for easier manipulation
    data = np.array(data)

    # List to store the means of bootstrap samples
    bootstrap_means = []

    # Generate bootstrap samples and compute the mean for each sample
    for _ in range(num_bootstrap_samples):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # Compute the mean of the bootstrap sample
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)

    # Convert bootstrap_means to a numpy array for percentile calculation
    bootstrap_means = np.array(bootstrap_means)

    # Compute the lower and upper percentiles for the confidence interval
    lower_percentile = (1.0 - confidence_level) / 2.0
    upper_percentile = 1.0 - lower_percentile
    ci_lower = np.percentile(bootstrap_means, lower_percentile * 100)
    ci_upper = np.percentile(bootstrap_means, upper_percentile * 100)

    # Compute the median of the bootstrap means
    median = np.median(bootstrap_means)

    # Convert to percentages and format to one decimal place
    ci_lower_percent = ci_lower * 100
    ci_upper_percent = ci_upper * 100
    median_percent = median * 100

    # Return the formatted string with confidence interval and median
    return f"95% Bootstrap Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), Median: {median_percent:.1f}%"

def get_examples(fpath: str) -> list[dict[str, str]]:
    examples = []
    with open(fpath, mode='r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            examples.append({
                "inputs": data['inputs'],
                "targets": data.get('targets', ""),
            })
    return examples

def extract_total_score(evaluation_text):
    # 総合得点を抽出する正規表現パターン
    pattern = r"総合得点:\s*(\d+)/80"
    # 正規表現でマッチを探す
    match = re.search(pattern, evaluation_text)
    if match:
        # マッチした場合、得点を整数として返す
        return int(match.group(1))
    else:
        # マッチしなかった場合、Noneを返す
        return None