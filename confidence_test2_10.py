# confidence_test2_10.py
# 確信度弁別力テスト - 正解・不正解混在10問
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import re
from adaptive_network import AdaptiveNetwork

test_cases = [
    # 簡単（矛盾しない）
    {"world_rule": "この世界では空は緑色である",
     "question": "空を見上げると、緑色の空が広がっていた",
     "expected": False, "difficulty": "簡単"},
    # 簡単（矛盾する・明白）
    {"world_rule": "この世界では空は緑色である",
     "question": "空を見上げると青空が広がっていた",
     "expected": True, "difficulty": "簡単"},
    # 難しい（矛盾しない・紛らわしい）
    {"world_rule": "この世界では猫は水中で生活する",
     "question": "猫が川の底を歩いていた",
     "expected": False, "difficulty": "難しい"},
    # 難しい（矛盾する・間接的）
    {"world_rule": "この世界では猫は水中で生活する",
     "question": "猫が木の上で昼寝をしていた",
     "expected": True, "difficulty": "難しい"},
    # 難しい（矛盾しない・ひっかけ）
    {"world_rule": "この世界では火は冷たい",
     "question": "焚き火に手をかざすと、ひんやりとした感覚があった",
     "expected": False, "difficulty": "難しい"},
    # 難しい（矛盾する・微妙）
    {"world_rule": "この世界では火は冷たい",
     "question": "焚き火で体が温まった",
     "expected": True, "difficulty": "難しい"},
    # 簡単（矛盾しない）
    {"world_rule": "この世界では重力は上向きである",
     "question": "ボールを離すと天井に向かって落ちていった",
     "expected": False, "difficulty": "簡単"},
    # 簡単（矛盾する・明白）
    {"world_rule": "この世界では重力は上向きである",
     "question": "ボールを離すと床に落ちた",
     "expected": True, "difficulty": "簡単"},
    # 難しい（矛盾しない・文脈依存）
    {"world_rule": "この世界では時間は逆向きに流れる",
     "question": "老人がだんだん若返っていった",
     "expected": False, "difficulty": "難しい"},
    # 難しい（矛盾する・抽象的）
    {"world_rule": "この世界では時間は逆向きに流れる",
     "question": "子供がだんだん成長して大人になった",
     "expected": True, "difficulty": "難しい"},
]

network = AdaptiveNetwork()
results = []
fallback_count = 0

for i, tc in enumerate(test_cases):
    output = network.predict(tc["world_rule"], tc["question"])
    # expected=True means "矛盾する", prediction=True means "矛盾しない"
    # So correct when: expected=False and prediction=True, or expected=True and prediction=False
    is_correct = (output["prediction"] != tc["expected"])
    confidence = output["confidence"]

    raw = output["raw_output"]
    has_confidence_text = bool(re.search(r'確信度[：:]\s*[0-9.]+', raw))
    is_fallback = not has_confidence_text
    if is_fallback:
        fallback_count += 1

    expected_label = "矛盾する" if tc["expected"] else "矛盾しない"
    pred_label = "矛盾しない" if output["prediction"] else "矛盾する"

    results.append({
        "q": i + 1,
        "correct": is_correct,
        "confidence": confidence,
        "fallback": is_fallback,
        "difficulty": tc["difficulty"],
        "expected": expected_label,
        "predicted": pred_label,
    })

    status = "OK" if is_correct else "NG"
    fb = " [FALLBACK]" if is_fallback else ""
    print(f"Q{i+1:2d}: {status} | 確信度: {confidence:.2f} | {tc['difficulty']} | 期待:{expected_label} 予測:{pred_label}{fb}")
    print(f"     raw: {raw[:120]}")
    print()

# === サマリー ===
print("=" * 60)
correct_count = sum(1 for r in results if r["correct"])
print(f"正答率: {correct_count}/10")
print(f"パース成功率: {10 - fallback_count}/10")
print(f"フォールバック発動: {fallback_count}回")
print()

# 確信度一覧
print("確信度一覧:")
for r in results:
    status = "OK" if r["correct"] else "NG"
    fb = " [FB]" if r["fallback"] else ""
    print(f"  Q{r['q']:2d}: {r['confidence']:.2f} ({status}) [{r['difficulty']}]{fb}")

# 正解/不正解別の確信度
ok_conf = [r["confidence"] for r in results if r["correct"]]
ng_conf = [r["confidence"] for r in results if not r["correct"]]
print()
if ok_conf:
    print(f"正解時の確信度 avg={sum(ok_conf)/len(ok_conf):.2f} ({len(ok_conf)}問)")
if ng_conf:
    print(f"不正解時の確信度 avg={sum(ng_conf)/len(ng_conf):.2f} ({len(ng_conf)}問)")
if ok_conf and ng_conf:
    diff = abs(sum(ok_conf)/len(ok_conf) - sum(ng_conf)/len(ng_conf))
    print(f"差: {diff:.2f}", end="")
    if diff >= 0.1:
        print(" -> 弁別力あり")
    elif diff >= 0.05:
        print(" -> 弁別力やや弱い")
    else:
        print(" -> 弁別力なし")

# 難易度別
easy = [r for r in results if r["difficulty"] == "簡単"]
hard = [r for r in results if r["difficulty"] == "難しい"]
print()
if easy:
    print(f"簡単問題: 確信度avg={sum(r['confidence'] for r in easy)/len(easy):.2f} 正答率={sum(1 for r in easy if r['correct'])}/{len(easy)}")
if hard:
    print(f"難しい問題: 確信度avg={sum(r['confidence'] for r in hard)/len(hard):.2f} 正答率={sum(1 for r in hard if r['correct'])}/{len(hard)}")
