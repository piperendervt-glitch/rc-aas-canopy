"""
fool_rc.py
Fool_RC：RCの判断ログを読んで盲点・矛盾を指摘する
Phase 2：片方向通信（Fool_Human→Fool_RCは未実装）

制約：
  - RCの制御下に置かない
  - 状態を保持しない
  - 黙らせる権限をRCに与えない
"""

import adaptive_network
from adaptive_network import call_ollama

# P5実験：Fool_RCのみ7bモデルを使用
adaptive_network.MODEL = "qwen2.5:7b"


FOOL_SYSTEM_PROMPT = """You are a Fool (jester).
Read the RC judgment log and ruthlessly criticize it.
Rules:
- NO summaries. Only criticism.
- Find contradictions, biases, blind spots.
- Output at least 3 specific criticisms.
- Respond in Japanese.
"""


def laugh_at_rc(log_text: str) -> str:
    """
    RCの判断ログを読んでFoolが指摘を返す。
    出力はそのまま（加工しない）Scribeに渡す。
    """
    prompt = f"""以下はRCの判断ログです。
矛盾・偏り・見落としを遠慮なく指摘してください。

--- ログ開始 ---
{log_text}
--- ログ終了 ---
"""
    raw_output = call_ollama(prompt, FOOL_SYSTEM_PROMPT)
    return raw_output  # 加工しない・原文のまま返す


if __name__ == "__main__":
    import sys
    from pathlib import Path

    log_file = Path("smoke_test_100_v7_output.txt")
    if not log_file.exists():
        print("ログファイルが見つかりません")
        sys.exit(1)

    try:
        log_text = log_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        log_text = log_file.read_text(encoding="cp932")
    # 最初の100行だけ使う（長すぎるとLLMが処理できない）
    log_lines = log_text.splitlines()[:100]
    log_excerpt = "\n".join(log_lines)

    print("=== Fool_RC の指摘 ===")
    output = laugh_at_rc(log_excerpt)
    print(output)
