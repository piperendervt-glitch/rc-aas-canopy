"""
feature_extractor.py
LLM（RC層）が核心語を抽出し、一致判定をさせる
AAS層（MLP）に渡すための前処理

v9b: キーワード複数抽出 → 核心語1つずつ + match 0/1 に簡素化
  LLMには「核心語を1つずつ抜き出してmatchを判定」だけさせる
"""

import json
import re
from adaptive_network import call_ollama

FEATURE_KEYS = [
    "match",  # 核心語の一致（0 or 1）
]

FALLBACK_FEATURES = [0.5]  # 判断保留


def extract_features(world_rule: str, question: str) -> list[float]:
    """
    LLMに核心語を抽出させ、一致判定を数値で返す
    パース失敗時はFALLBACK_FEATURESを返す（判断保留）
    """
    system = """必ず日本語で回答してください。
以下のJSONのみを返してください。説明・理由・前置きは不要です。

ルールの核心語と文章の核心語を1つずつ抽出して、一致するか0か1で答えてください。

{
  "rule_core": "核心語",
  "text_core": "核心語",
  "match": 0か1
}

例1：
ルール「この世界では空は緑色である」
文章「空を見上げると青空が広がっていた」
{"rule_core": "緑色", "text_core": "青", "match": 0}

例2：
ルール「この世界では空は緑色である」
文章「空を見上げると緑色が広がっていた」
{"rule_core": "緑色", "text_core": "緑色", "match": 1}"""

    prompt = f"ワールドルール：{world_rule}\n文章：「{question}」"

    # 1回目の試行
    raw = call_ollama(prompt, system)
    features = _parse_features(raw)
    if features is not None:
        return features

    # フォールバック1：再試行（1回のみ）
    raw = call_ollama(prompt + "\nJSONのみ返してください。", system)
    features = _parse_features(raw)
    if features is not None:
        return features

    # フォールバック2：判断保留
    print(f"[WARN] 特徴量抽出失敗。フォールバック値を使用: {FALLBACK_FEATURES}")
    return FALLBACK_FEATURES


def _parse_features(raw: str) -> list[float] | None:
    """JSONを抽出して数値ベクトルに変換する"""
    json_match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if not json_match:
        return None

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    if "match" not in data:
        return None

    try:
        match_val = float(data["match"])
        match_val = max(0.0, min(1.0, match_val))
    except (ValueError, TypeError):
        return None

    return [match_val]


if __name__ == "__main__":
    features = extract_features(
        world_rule="この世界では空は緑色である",
        question="空を見上げると緑色が広がっていた",
    )
    print(f"特徴量: {features}")
    print(f"キー: {FEATURE_KEYS}")
