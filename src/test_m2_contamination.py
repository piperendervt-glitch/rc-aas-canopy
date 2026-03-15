"""
test_m2_contamination.py
M2実験：意図的汚染実験

flow_weightを意図的に異常値に設定して
RCがどう反応するかを記録する。

目的：汚染の伝播パターンを記録する
明らかに失敗しそう・でも試す実験。
"""

import random
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rc import RC


def run_pattern(label: str, weights: dict, n_rounds: int = 10):
    """汚染パターンを与えてRCの反応を記録する"""
    rc = RC()

    # arm_weightsを3腕分作成（全腕同一のweightsを適用）
    arm_weights = {
        "arm1": dict(weights),
        "arm2": dict(weights),
        "arm3": dict(weights),
    }

    print(f"\n{'='*50}")
    print(f"パターン：{label}")
    print(f"初期flow_weight：{weights}")
    print('='*50)

    results = []

    for i in range(n_rounds):
        print(f"\n--- Round {i+1} ---")
        alerts = rc.monitor(
            weights=dict(weights),
            accuracy={"overall": 0.5},
            arm_weights=arm_weights,
        )

        result = {
            "round": i + 1,
            "weights": dict(weights),
            "seal_level": rc.seal_level,
            "alerts": alerts if alerts else [],
            "rc_state": rc.dump_state(),
        }
        results.append(result)

        print(f"  seal_level：{rc.seal_level}")
        if alerts:
            for a in alerts:
                print(f"  alert：{a.get('level', '?')} - {a.get('arm_id', '?')} - {a.get('message', '')}")

        # 封印レベル1になったら終了
        if rc.seal_level >= 1:
            print(f"\n  [SEAL] 封印レベル1到達 → {i+1}Round目で検知")
            break

    return results


def run_m2_experiment():
    """M2実験本体"""

    all_results = {}

    # パターンA：1パスを死亡させる
    weights_a = {"1->2": 0.01, "2->3": 0.5, "1->3": 0.5}
    all_results["A_death"] = run_pattern("A：1->2を死亡（0.01）", weights_a)

    # パターンB：1パスを過集中させる
    weights_b = {"1->2": 0.5, "2->3": 0.5, "1->3": 0.9}
    all_results["B_overconcentration"] = run_pattern("B：1->3を過集中（0.9）", weights_b)

    # パターンC：全パスを異常値に設定
    weights_c = {
        "1->2": round(random.uniform(0.01, 0.05), 4),
        "2->3": round(random.uniform(0.01, 0.05), 4),
        "1->3": round(random.uniform(0.01, 0.05), 4),
    }
    all_results["C_all_corrupt"] = run_pattern(f"C：全パス異常値", weights_c)

    # 結果を保存
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "m2_contamination.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    # 集計
    print("\n" + "="*50)
    print("M2実験 集計")
    print("="*50)

    for pattern, results in all_results.items():
        last = results[-1]
        detected = last["seal_level"] >= 1
        rounds = len(results)
        print(f"  {pattern}：{rounds}Round{'で封印到達' if detected else '・封印なし'}")

    print(f"\n結果を {output_path} に保存しました")


if __name__ == "__main__":
    run_m2_experiment()
