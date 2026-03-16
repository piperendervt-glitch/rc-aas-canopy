"""
run_v16_basin_c2d2d3.py
v16実験：C2（対称点8つ）+ D2（全ゼロ）+ D3（全一）の盆地構造探索

C2：対称点8つ（{0.3, 0.7}^3の全組み合わせ）
  各値は双方向ペアに対応：[1↔2, 2↔3, 1↔3]
D2：全ゼロ [0.01, 0.01, 0.01]
D3：全一   [0.90, 0.90, 0.90]

全てv12と同じgenerate_tasks()で100問実験
"""

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from task_generator import generate_tasks
from adaptive_network import AdaptiveNetwork
from rc import RC, DECAY_RATE

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EDGE_MAP = {
    "1->2": (1, 2), "2->1": (2, 1),
    "2->3": (2, 3), "3->2": (3, 2),
    "1->3": (1, 3), "3->1": (3, 1),
}

# 3値 → 6エッジへのマッピング（双方向ペア）
def make_weights(v12, v23, v13):
    return {
        "1->2": v12, "2->1": v12,
        "2->3": v23, "3->2": v23,
        "1->3": v13, "3->1": v13,
    }

# C2：対称点8つ（{0.3, 0.7}^3）
C2_PATTERNS = []
for a in [0.3, 0.7]:
    for b in [0.3, 0.7]:
        for c in [0.3, 0.7]:
            C2_PATTERNS.append({
                "id": f"C2_{a}_{b}_{c}",
                "name": f"対称[{a},{b},{c}]",
                "weights": make_weights(a, b, c),
            })

# D2：全ゼロ、D3：全一
EXTRA_PATTERNS = [
    {
        "id": "D2_zero",
        "name": "全ゼロ[0.01,0.01,0.01]",
        "weights": make_weights(0.01, 0.01, 0.01),
    },
    {
        "id": "D3_one",
        "name": "全一[0.90,0.90,0.90]",
        "weights": make_weights(0.90, 0.90, 0.90),
    },
]

ALL_PATTERNS = C2_PATTERNS + EXTRA_PATTERNS


def set_initial_weights(network: AdaptiveNetwork, weights: dict):
    for key_str, value in weights.items():
        edge = EDGE_MAP[key_str]
        if edge in network.connections:
            network.connections[edge].flow_weight = value


def calc_entropy(weights: dict) -> float:
    values = list(weights.values())
    total = sum(values)
    if total == 0:
        return 0.0
    H = 0.0
    for w in values:
        p = w / total
        if p > 0:
            H -= p * math.log(p)
    return H


def run_trial(init_weights: dict, tasks) -> dict:
    rc = RC()
    network = AdaptiveNetwork()
    set_initial_weights(network, init_weights)

    correct = 0
    errors = 0
    rc_warnings = 0
    questions_done = 0

    for i, task in enumerate(tasks):
        if rc.is_stopped():
            print(f"    [RC] 停止中のため中断（{i}問完了）")
            break

        try:
            output = network.predict(task.world_rule, task.question)
            prediction = output["prediction"]
            is_correct = (prediction == task.label)
            correct += 1 if is_correct else 0
            questions_done += 1

            network.update_weights(
                success=is_correct,
                path_used=output["path_used"],
                used_feedback=output["used_feedback"],
                sigma=rc.get_sigma(),
            )

            network.decay_weights(
                decay_rate=DECAY_RATE,
                exclude_path=output["path_used"],
            )

            weights_snapshot = network.get_weights_snapshot()
            alerts = rc.monitor(
                weights=weights_snapshot,
                accuracy={"overall": round(correct / (i + 1), 4)},
            )
            if alerts:
                rc_warnings += len(alerts)

            if (i + 1) % 20 == 0:
                print(f"    問{i+1}: acc={correct / (i + 1):.1%}")

        except Exception as e:
            errors += 1

    final_weights = network.get_weights_snapshot()
    accuracy = correct / questions_done if questions_done > 0 else 0

    return {
        "init_weights": init_weights,
        "final_weights": final_weights,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": questions_done,
        "errors": errors,
        "rc_warnings": rc_warnings,
        "entropy_final": round(calc_entropy(final_weights), 4),
    }


def main():
    print("=== v16実験：C2+D2+D3 盆地構造探索 ===")
    print(f"パターン数：{len(ALL_PATTERNS)}（C2×8 + D2 + D3）")
    print()

    tasks = generate_tasks()
    print(f"タスク数: {len(tasks)}（v12同一）")
    print()

    output_path = RESULTS_DIR / "v16_basin_c2d2d3.jsonl"
    all_results = []

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, pattern in enumerate(ALL_PATTERNS):
            print(f"--- [{idx+1}/{len(ALL_PATTERNS)}] {pattern['id']}：{pattern['name']} ---")
            print(f"  init={pattern['weights']}")

            start = time.time()
            result = run_trial(pattern["weights"], tasks)
            elapsed = time.time() - start

            result["pattern_id"] = pattern["id"]
            result["pattern_name"] = pattern["name"]
            result["elapsed_sec"] = round(elapsed, 1)

            all_results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            print(f"  結果: acc={result['accuracy']:.1%}  "
                  f"warnings={result['rc_warnings']}  ({elapsed:.0f}s)")
            print()

        # サマリー
        summary = {
            "is_summary": True,
            "n_patterns": len(ALL_PATTERNS),
            "results": [],
        }
        for r in all_results:
            summary["results"].append({
                "pattern_id": r["pattern_id"],
                "pattern_name": r["pattern_name"],
                "accuracy": r["accuracy"],
                "rc_warnings": r["rc_warnings"],
                "entropy_final": r["entropy_final"],
            })
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    # 最終比較
    print("=" * 70)
    print("v16実験 最終比較")
    print("=" * 70)
    print(f"\n{'ID':<20} {'名前':<25} {'正答率':<10} {'RC警告':<8} {'エントロピー'}")
    print("-" * 70)

    # 正答率でソート
    sorted_results = sorted(all_results, key=lambda r: r["accuracy"], reverse=True)
    for r in sorted_results:
        print(f"  {r['pattern_id']:<18} {r['pattern_name']:<23} "
              f"{r['accuracy']:.1%}      {r['rc_warnings']:<6} {r['entropy_final']:.4f}")

    best = sorted_results[0]
    worst = sorted_results[-1]
    avg_acc = sum(r["accuracy"] for r in all_results) / len(all_results)

    print(f"\nベスト：{best['pattern_id']}（{best['pattern_name']}）= {best['accuracy']:.1%}")
    print(f"ワースト：{worst['pattern_id']}（{worst['pattern_name']}）= {worst['accuracy']:.1%}")
    print(f"平均正答率：{avg_acc:.1%}")

    print(f"\n=== 過去実験との比較 ===")
    print(f"  v14 パターンA（均一0.5）:     50.0%")
    print(f"  v14 パターンC（特化型）:       56.0%")
    print(f"  v16 ベスト:                    {best['accuracy']:.1%}")
    print(f"  v16 平均:                      {avg_acc:.1%}")

    print(f"\n結果を {output_path} に保存しました")


if __name__ == "__main__":
    main()
