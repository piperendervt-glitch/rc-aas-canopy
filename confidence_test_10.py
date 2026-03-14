# confidence_test_10.py
# 確信度（自己申告）の動作確認テスト - 10問
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from task_generator import generate_tasks
from adaptive_network import AdaptiveNetwork

tasks = generate_tasks()[:10]  # 最初の10問
network = AdaptiveNetwork()

results = []
fallback_count = 0

for i, task in enumerate(tasks):
    output = network.predict(task.world_rule, task.question)
    is_correct = (output["prediction"] == task.label)
    confidence = output["confidence"]

    # フォールバック判定（0.5はパース失敗時のデフォルト）
    raw = output["raw_output"]
    import re
    has_confidence_text = bool(re.search(r'確信度[：:]\s*[0-9.]+', raw))
    is_fallback = not has_confidence_text

    if is_fallback:
        fallback_count += 1

    results.append({
        "q": i + 1,
        "correct": is_correct,
        "confidence": confidence,
        "fallback": is_fallback,
        "label": task.label,
        "prediction": output["prediction"],
    })

    status = "OK" if is_correct else "NG"
    fb = " [FALLBACK]" if is_fallback else ""
    print(f"[{i+1:2d}] {status} | 確信度: {confidence:.2f}{fb}")
    print(f"     raw: {raw[:100]}")
    print()

# サマリー
print("=" * 60)
print(f"パース成功率: {10 - fallback_count}/10")
print(f"フォールバック発動: {fallback_count}回")
print()

# 確信度一覧
print("確信度一覧:")
for r in results:
    status = "OK" if r["correct"] else "NG"
    print(f"  Q{r['q']:2d}: {r['confidence']:.2f} ({status})")

# 分布確認
confidences = [r["confidence"] for r in results]
print(f"\n確信度 min={min(confidences):.2f} max={max(confidences):.2f} avg={sum(confidences)/len(confidences):.2f}")

# 正解/不正解別
ok_conf = [r["confidence"] for r in results if r["correct"]]
ng_conf = [r["confidence"] for r in results if not r["correct"]]
if ok_conf:
    print(f"正解時の確信度 avg={sum(ok_conf)/len(ok_conf):.2f} ({len(ok_conf)}問)")
if ng_conf:
    print(f"不正解時の確信度 avg={sum(ng_conf)/len(ng_conf):.2f} ({len(ng_conf)}問)")

# 全部同じ値チェック
unique_vals = set(round(c, 2) for c in confidences)
if len(unique_vals) == 1:
    print(f"\n⚠ 確信度が全問同一値: {unique_vals.pop()}")
else:
    print(f"\n確信度のユニーク値数: {len(unique_vals)}")
