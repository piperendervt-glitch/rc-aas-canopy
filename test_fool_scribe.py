import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fool_rc import laugh_at_rc
from scribe import record

# v7のログを読み込む
log_file = Path("smoke_test_100_v7_output.txt")
try:
    log_text = log_file.read_text(encoding="utf-8")
except UnicodeDecodeError:
    log_text = log_file.read_text(encoding="cp932")

log_excerpt = "\n".join(log_text.splitlines()[:100])

# Foolが笑う
fool_output = laugh_at_rc(log_excerpt)
print("=== Fool_RC の指摘 ===")
print(fool_output)

# Scribeが記録する
record(fool_output, input_description="smoke_test_100_v7 Q1-100（先頭100行）")
print("\n=== Scribe 記録完了 ===")
