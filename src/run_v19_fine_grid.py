"""
run_v19_fine_grid.py
v19実験（②）：[0.3,0.7,0.7]周辺の細かいグリッド

v16ベスト周辺を0.1刻みオフセットグリッドで探索
  1↔2: {0.15, 0.25, 0.35, 0.45} （0.3中心 ±0.15）
  2↔3: {0.55, 0.65, 0.75, 0.85} （0.7中心 ±0.15）
  1↔3: {0.55, 0.65, 0.75, 0.85} （0.7中心 ±0.15）
= 4^3 = 64点、v12と同じgenerate_tasks()で100問
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

GRID_V12 = [0.15, 0.25, 0.35, 0.45]
GRID_V23 = [0.55, 0.65, 0.75, 0.85]
GRID_V13 = [0.55, 0.65, 0.75, 0.85]


def make_weights(v12, v23, v13):
    return {
        "1->2": v12, "2->1": v12,
        "2->3": v23, "3->2": v23,
        "1->3": v13, "3->1": v13,
    }


def set_initial_weights(network, weights):
    for key_str, value in weights.items():
        edge = EDGE_MAP[key_str]
        if edge in network.connections:
            network.connections[edge].flow_weight = value


def calc_entropy(weights):
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


def run_trial(init_weights, tasks):
    rc = RC()
    network = AdaptiveNetwork()
    set_initial_weights(network, init_weights)

    correct = 0
    errors = 0
    rc_warnings = 0
    questions_done = 0

    for i, task in enumerate(tasks):
        if rc.is_stopped():
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
    # グリッド生成
    patterns = []
    for v12 in GRID_V12:
        for v23 in GRID_V23:
            for v13 in GRID_V13:
                patterns.append({
                    "id": f"F_{v12}_{v23}_{v13}",
                    "v12": v12, "v23": v23, "v13": v13,
                    "weights": make_weights(v12, v23, v13),
                })

    print(f"=== v19実験：[0.3,0.7,0.7]周辺 細かいグリッド ===")
    print(f"1-2: {GRID_V12}")
    print(f"2-3: {GRID_V23}")
    print(f"1-3: {GRID_V13}")
    print(f"合計: {len(patterns)}点")
    print()

    tasks = generate_tasks()
    print(f"タスク数: {len(tasks)}（v12同一）")
    print()

    output_path = RESULTS_DIR / "v19_fine_grid.jsonl"
    all_results = []
    start_all = time.time()

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, pattern in enumerate(patterns):
            start = time.time()
            result = run_trial(pattern["weights"], tasks)
            elapsed = time.time() - start

            result["pattern_id"] = pattern["id"]
            result["v12"] = pattern["v12"]
            result["v23"] = pattern["v23"]
            result["v13"] = pattern["v13"]
            result["elapsed_sec"] = round(elapsed, 1)

            all_results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            if (idx + 1) % 8 == 0:
                eta = (time.time() - start_all) / (idx + 1) * (len(patterns) - idx - 1)
                print(f"  [{idx+1}/{len(patterns)}] {pattern['id']}: "
                      f"acc={result['accuracy']:.0%}  (ETA {eta/60:.0f}min)")

        # サマリー
        sorted_results = sorted(all_results, key=lambda r: r["accuracy"], reverse=True)
        summary = {
            "is_summary": True,
            "n_patterns": len(patterns),
            "grid_v12": GRID_V12,
            "grid_v23": GRID_V23,
            "grid_v13": GRID_V13,
            "results": [
                {"id": r["pattern_id"], "v12": r["v12"], "v23": r["v23"], "v13": r["v13"],
                 "accuracy": r["accuracy"], "rc_warnings": r["rc_warnings"]}
                for r in sorted_results
            ],
        }
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    total_time = time.time() - start_all

    print()
    print("=" * 70)
    print("v19実験 細かいグリッド結果")
    print("=" * 70)

    print(f"\n--- 上位10 ---")
    print(f"  {'ID':<22} {'1-2':<6} {'2-3':<6} {'1-3':<6} {'acc':<8} {'warn'}")
    print("-" * 60)
    for r in sorted_results[:10]:
        print(f"  {r['pattern_id']:<20} {r['v12']:<6} {r['v23']:<6} {r['v13']:<6} "
              f"{r['accuracy']:.0%}      {r['rc_warnings']}")

    print(f"\n--- 下位5 ---")
    for r in sorted_results[-5:]:
        print(f"  {r['pattern_id']:<20} {r['v12']:<6} {r['v23']:<6} {r['v13']:<6} "
              f"{r['accuracy']:.0%}      {r['rc_warnings']}")

    best = sorted_results[0]
    accs = [r["accuracy"] for r in all_results]
    avg_acc = sum(accs) / len(accs)

    print(f"\n全{len(patterns)}点 平均: {avg_acc:.1%}")
    print(f"ベスト: {best['pattern_id']} [{best['v12']},{best['v23']},{best['v13']}] = {best['accuracy']:.0%}")
    print(f"v16ベスト比較: [0.3,0.7,0.7] = 62%")
    print(f"総実行時間: {total_time/60:.1f}分")
    print(f"\n結果を {output_path} に保存しました")


if __name__ == "__main__":
    main()
