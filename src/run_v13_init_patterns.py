"""
run_v13_init_patterns.py
v13実験：flow_weight初期値パターンB〜Eの比較実験

パターンA：均一（0.5, 0.5, 0.5）= v12ベースライン 69%
パターンB：ランダム（3回試行平均）
パターンC：タスク特化型（1->2=0.7, 2->3=0.5, 1->3=0.3）
パターンD：過去の成功パターン（G1 loop3 acc=90%時のweights）
パターンE：多様性強制（1->2=0.3, 2->3=0.5, 1->3=0.7）

各パターンで100問実験を実行。
結果はresults/v13_init_patterns.jsonlに保存。
"""

import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from task_generator import generate_tasks
from adaptive_network import AdaptiveNetwork, Connection, INITIAL_WEIGHT
from rc import RC, DECAY_RATE

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_QUESTIONS = 100

# パターンD：G1 loop3（最高正答率90%）時のweights
PATTERN_D_WEIGHTS = {
    "1->2": 0.5994, "2->1": 0.4522,
    "2->3": 0.8955, "3->2": 0.4522,
    "1->3": 0.2644, "3->1": 0.4522,
}

PATTERNS = {
    "A": {
        "name": "均一（ベースライン）",
        "weights": {
            "1->2": 0.5, "2->1": 0.5,
            "2->3": 0.5, "3->2": 0.5,
            "1->3": 0.5, "3->1": 0.5,
        },
        "trials": 1,
    },
    "B": {
        "name": "ランダム",
        "weights": None,  # 毎回乱数で生成
        "trials": 3,
    },
    "C": {
        "name": "タスク特化型",
        "weights": {
            "1->2": 0.7, "2->1": 0.5,
            "2->3": 0.5, "3->2": 0.5,
            "1->3": 0.3, "3->1": 0.5,
        },
        "trials": 1,
    },
    "D": {
        "name": "過去の成功パターン（G1 loop3 acc=90%）",
        "weights": PATTERN_D_WEIGHTS,
        "trials": 1,
    },
    "E": {
        "name": "多様性強制",
        "weights": {
            "1->2": 0.3, "2->1": 0.5,
            "2->3": 0.5, "3->2": 0.5,
            "1->3": 0.7, "3->1": 0.5,
        },
        "trials": 1,
    },
}

# エッジキーとタプルの対応
EDGE_MAP = {
    "1->2": (1, 2), "2->1": (2, 1),
    "2->3": (2, 3), "3->2": (3, 2),
    "1->3": (1, 3), "3->1": (3, 1),
}


def set_initial_weights(network: AdaptiveNetwork, weights: dict):
    """ネットワークのflow_weightを指定値に設定"""
    for key_str, value in weights.items():
        edge = EDGE_MAP[key_str]
        if edge in network.connections:
            network.connections[edge].flow_weight = value


def generate_random_weights(seed: int) -> dict:
    """ランダム初期値を生成"""
    rng = random.Random(seed)
    paths = ["1->2", "2->1", "2->3", "3->2", "1->3", "3->1"]
    return {p: round(rng.uniform(0.1, 0.9), 4) for p in paths}


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


def run_single_trial(init_weights: dict, tasks, trial_id: int = 0) -> dict:
    """1回の試行（100問）を実行"""
    rc = RC()
    network = AdaptiveNetwork()
    set_initial_weights(network, init_weights)

    correct = 0
    errors = 0
    rc_warnings = 0
    questions_done = 0

    for i, task in enumerate(tasks[:N_QUESTIONS]):
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

            status = "o" if is_correct else "x"
            if (i + 1) % 20 == 0:
                acc_so_far = correct / (i + 1)
                print(f"    問{i+1}: acc={acc_so_far:.1%}")

        except Exception as e:
            errors += 1
            if (i + 1) % 20 == 0:
                print(f"    問{i+1}: ERROR")

    final_weights = network.get_weights_snapshot()
    accuracy = correct / questions_done if questions_done > 0 else 0

    return {
        "trial": trial_id,
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
    print("=== v13実験：flow_weight初期値パターン比較 ===")
    print(f"各パターンで{N_QUESTIONS}問実行")
    print()

    tasks = generate_tasks(count=N_QUESTIONS, seed=42)

    output_path = RESULTS_DIR / "v13_init_patterns.jsonl"
    all_results = []

    with open(output_path, "w", encoding="utf-8") as f:
        for pattern_id in ["A", "B", "C", "D", "E"]:
            pattern = PATTERNS[pattern_id]
            n_trials = pattern["trials"]
            print(f"--- パターン{pattern_id}：{pattern['name']}（{n_trials}回試行） ---")

            trial_results = []
            for trial in range(n_trials):
                if pattern["weights"] is None:
                    # パターンB：ランダム
                    init_weights = generate_random_weights(seed=100 + trial)
                    print(f"  試行{trial+1}: init={init_weights}")
                else:
                    init_weights = pattern["weights"]
                    if n_trials == 1:
                        print(f"  init={init_weights}")

                start = time.time()
                result = run_single_trial(init_weights, tasks, trial_id=trial)
                elapsed = time.time() - start

                result["pattern"] = pattern_id
                result["pattern_name"] = pattern["name"]
                result["elapsed_sec"] = round(elapsed, 1)

                trial_results.append(result)
                print(f"  試行{trial+1}: acc={result['accuracy']:.1%}  "
                      f"warnings={result['rc_warnings']}  ({elapsed:.0f}s)")

                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

            # 平均集計
            avg_acc = sum(r["accuracy"] for r in trial_results) / len(trial_results)
            avg_warnings = sum(r["rc_warnings"] for r in trial_results) / len(trial_results)
            summary = {
                "pattern": pattern_id,
                "pattern_name": pattern["name"],
                "avg_accuracy": round(avg_acc, 4),
                "avg_warnings": round(avg_warnings, 1),
                "n_trials": n_trials,
                "is_summary": True,
            }
            all_results.append(summary)
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
            f.flush()

            print(f"  → 平均正答率: {avg_acc:.1%}\n")

    # 最終比較
    print("=" * 60)
    print("v13実験 最終比較")
    print("=" * 60)
    print(f"\n{'パターン':<12} {'名前':<30} {'正答率':<10} {'RC警告':<8}")
    print("-" * 60)
    for s in all_results:
        print(f"  {s['pattern']:<10} {s['pattern_name']:<28} {s['avg_accuracy']:.1%}      {s['avg_warnings']:.0f}")

    # v12との比較
    print(f"\n比較対象：パターンA（v12） = 69%（目標値）")
    best = max(all_results, key=lambda r: r["avg_accuracy"])
    worst = min(all_results, key=lambda r: r["avg_accuracy"])
    print(f"ベスト：パターン{best['pattern']}（{best['pattern_name']}）= {best['avg_accuracy']:.1%}")
    print(f"ワースト：パターン{worst['pattern']}（{worst['pattern_name']}）= {worst['avg_accuracy']:.1%}")

    print(f"\n結果を {output_path} に保存しました")


if __name__ == "__main__":
    main()
