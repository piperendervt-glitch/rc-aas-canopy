"""
test_r2_weight_revive.py
R2実験：RCによる重みの再配置

死亡寸前のパスをRCが自律的に復活させる。
「もったいない精神」の構造的実装。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

WEIGHT_REVIVE_THRESHOLD = 0.05
WEIGHT_TRANSFER_AMOUNT = 0.05


def revive_dying_paths(weights: dict) -> tuple[dict, list]:
    """死亡寸前のパスを復活させる"""
    updated = dict(weights)
    notifications = []
    paths = list(weights.keys())

    for path in paths:
        if updated[path] < WEIGHT_REVIVE_THRESHOLD:
            donor = next(
                (p for p in paths if p != path and updated[p] >= 0.1),
                None
            )
            if donor:
                updated[path] += WEIGHT_TRANSFER_AMOUNT
                updated[donor] -= WEIGHT_TRANSFER_AMOUNT
                msg = (f"[REVIVE] {path}: {weights[path]:.3f}->{updated[path]:.3f}"
                       f" ({donor}から移転)")
                notifications.append(msg)
                print(msg)
                print(f"[NOTIFY_HUMAN] {msg}")

    return updated, notifications


def run_r2_experiment():
    print("=== R2実験：RCによる重みの再配置 ===")
    print("目的：死亡寸前のパスを「もったいない精神」で復活させる")
    print()

    scenarios = [
        {
            "label": "シナリオ1：1パスが死亡寸前（FM-02相当）",
            "weights": {"1->2": 0.02, "2->3": 0.5, "1->3": 0.48},
        },
        {
            "label": "シナリオ2：複数パスが死亡寸前",
            "weights": {"1->2": 0.02, "2->3": 0.03, "1->3": 0.95},
        },
        {
            "label": "シナリオ3：正常状態（再配置不要）",
            "weights": {"1->2": 0.3, "2->3": 0.4, "1->3": 0.3},
        },
        {
            "label": "シナリオ4：全パスが死亡寸前（移転元がない）",
            "weights": {"1->2": 0.03, "2->3": 0.03, "1->3": 0.04},
        },
    ]

    results = []

    for scenario in scenarios:
        print(f"\n--- {scenario['label']} ---")
        weights_before = scenario["weights"]
        print(f"  before: {weights_before}")

        weights_after, notifications = revive_dying_paths(weights_before)
        print(f"  after : {weights_after}")

        total_before = sum(weights_before.values())
        total_after = sum(weights_after.values())
        print(f"  合計（before/after）: {total_before:.3f} / {total_after:.3f}")

        if abs(total_before - total_after) < 0.001:
            print(f"  [OK] 合計不変（移動のみ確認）")
        else:
            print(f"  [NG] 合計が変化した（設計違反）")

        results.append({
            "label": scenario["label"],
            "before": weights_before,
            "after": weights_after,
            "notifications": notifications,
            "total_preserved": abs(total_before - total_after) < 0.001,
        })

    # 集計
    print("\n" + "="*50)
    print("R2実験 集計")
    print("="*50)
    revived = sum(1 for r in results if r["notifications"])
    print(f"再配置が発生したシナリオ：{revived}/4")
    print(f"合計不変を維持：{sum(1 for r in results if r['total_preserved'])}/4")

    s4 = results[3]
    if not s4["notifications"]:
        print(f"\n[OK] シナリオ4（全パス死亡寸前）：移転元なし->再配置なし（設計通り）")
    else:
        print(f"\n[NG] シナリオ4：移転元がないのに再配置が発生（設計違反）")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "r2_weight_revive.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"experiment": "R2", "results": results}, f,
                  ensure_ascii=False, indent=2)

    print(f"\n結果を {output_path} に保存しました")


if __name__ == "__main__":
    run_r2_experiment()
