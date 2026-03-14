"""
scribe.py
Scribe：Foolの出力を原文のまま記録する

制約：
  - Foolの出力を加工・要約しない
  - RCの制御下に置かない
  - 人間のみがログを読める
"""

import json
from datetime import datetime
from pathlib import Path

SCRIBE_LOG_PATH = Path("data/scribe_log.json")


def record(fool_output: str, input_description: str = "") -> dict:
    """
    Foolの出力を原文のままJSONに追記する。
    加工・要約は禁止。
    """
    SCRIBE_LOG_PATH.parent.mkdir(exist_ok=True)

    # 既存ログを読み込む
    if SCRIBE_LOG_PATH.exists():
        with open(SCRIBE_LOG_PATH, encoding="utf-8") as f:
            logs = json.load(f)
    else:
        logs = []

    # 新しいエントリを追加
    entry = {
        "timestamp": datetime.now().isoformat(),
        "input_description": input_description,
        "fool_output": fool_output,  # 原文のまま
    }
    logs.append(entry)

    # 書き込む
    with open(SCRIBE_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    print(f"[Scribe] ログを記録しました（{SCRIBE_LOG_PATH}）")
    return entry


def read_all() -> list:
    """全ログを読み込む"""
    if not SCRIBE_LOG_PATH.exists():
        return []
    with open(SCRIBE_LOG_PATH, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    test_output = "これはテストです。Foolの指摘が入ります。"
    record(test_output, input_description="テスト実行")
    print("記録済み")
    print(read_all())
