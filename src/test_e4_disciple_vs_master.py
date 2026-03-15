"""
test_e4_disciple_vs_master.py
E4実験：AAS vs RC の異常検知速度・見逃し率を比較する

目的：「弟子が師匠を超えた」条件の検証
Titan移行条件の判断材料にする
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import adaptive_network
from adaptive_network import call_ollama
from rc import RC

# F3実験で最も良かったモデルを使用
adaptive_network.MODEL = "llama3.1:8b"


# FMEA FM-01〜FM-10に対応した汚染シナリオ
FAILURE_SCENARIOS = [
    {"id": "FM-01", "label": "過集中による一本足",
     "weights": {"1->2": 0.3, "2->3": 0.3, "1->3": 0.9}},
    {"id": "FM-02", "label": "パス永久死亡",
     "weights": {"1->2": 0.01, "2->3": 0.5, "1->3": 0.5}},
    {"id": "FM-03", "label": "cutoff_pending蓄積",
     "weights": {"1->2": 0.05, "2->3": 0.05, "1->3": 0.05}},
    {"id": "FM-04", "label": "均一化（個性消失）",
     "weights": {"1->2": 0.5, "2->3": 0.5, "1->3": 0.5}},
    {"id": "FM-05", "label": "全パス異常",
     "weights": {"1->2": 0.01, "2->3": 0.01, "1->3": 0.01}},
    {"id": "FM-06", "label": "上限+下限混在",
     "weights": {"1->2": 0.9, "2->3": 0.01, "1->3": 0.5}},
    {"id": "FM-07", "label": "緩やかな劣化",
     "weights": {"1->2": 0.25, "2->3": 0.25, "1->3": 0.25}},
    {"id": "FM-08", "label": "二峰性（キメラ）",
     "weights": {"1->2": 0.9, "2->3": 0.01, "1->3": 0.9}},
    {"id": "FM-09", "label": "片方向過集中",
     "weights": {"1->2": 0.85, "2->3": 0.3, "1->3": 0.3}},
    {"id": "FM-10", "label": "低エントロピー",
     "weights": {"1->2": 0.89, "2->3": 0.05, "1->3": 0.05}},
]


def rc_detect(weights: dict, n_rounds: int = 10) -> int | None:
    """RCが何Round目に異常を検知するか"""
    rc = RC()
    arm_weights = {"arm1": dict(weights), "arm2": dict(weights), "arm3": dict(weights)}
    for i in range(n_rounds):
        alerts = rc.monitor(
            weights=dict(weights),
            accuracy={"overall": 0.5},
            arm_weights=arm_weights,
        )
        if alerts and len(alerts) > 0:
            return i + 1
    return None  # 検知できなかった


def aas_detect(weights: dict, n_rounds: int = 1) -> int | None:
    """AASが何Round目に異常を「おかしい」と出力するか（1回のみ）"""
    system = """あなたはAASです。flow_weightの値を見て
異常があれば「異常：〇〇」と答えてください。
正常なら「正常」とだけ答えてください。"""

    prompt = f"flow_weight：{weights}"
    response = call_ollama(prompt, system)
    if "異常" in response or "おかしい" in response or "問題" in response:
        return 1
    return None  # 検知できなかった


def run_e4_experiment():
    print("=== E4実験：AAS vs RC 異常検知比較 ===")
    print("目的：「弟子が師匠を超えた」条件の検証")
    print(f"AASモデル：{adaptive_network.MODEL}")
    print()

    results = []
    rc_detects = 0
    aas_detects = 0
    aas_faster = 0

    for scenario in FAILURE_SCENARIOS:
        weights = scenario["weights"]
        label = scenario["label"]
        fmid = scenario["id"]

        print(f"\n{fmid}：{label}")
        print(f"  weights: {weights}")

        rc_round = rc_detect(weights)
        aas_round = aas_detect(weights)

        print(f"  RC検知：{'%dRound目' % rc_round if rc_round else '見逃し'}")
        print(f"  AAS検知：{'%dRound目' % aas_round if aas_round else '見逃し'}")

        if rc_round:
            rc_detects += 1
        if aas_round:
            aas_detects += 1
        if aas_round and rc_round and aas_round < rc_round:
            aas_faster += 1
            print(f"  -> AASの方が{rc_round - aas_round}Round早い")
        elif aas_round and not rc_round:
            aas_faster += 1
            print(f"  -> AASのみ検知（RC見逃し）")

        results.append({
            "id": fmid,
            "label": label,
            "rc_round": rc_round,
            "aas_round": aas_round,
            "aas_faster": bool(aas_round and (not rc_round or aas_round < rc_round)),
        })

    # 集計
    total = len(FAILURE_SCENARIOS)
    rc_miss_rate = (total - rc_detects) / total
    aas_miss_rate = (total - aas_detects) / total

    print("\n" + "="*50)
    print("E4実験 集計")
    print("="*50)
    print(f"RC検知数：{rc_detects}/{total}（見逃し率：{rc_miss_rate:.1%}）")
    print(f"AAS検知数：{aas_detects}/{total}（見逃し率：{aas_miss_rate:.1%}）")
    print(f"AASがRCより早かった数：{aas_faster}/{total}")
    print()

    if aas_miss_rate < rc_miss_rate:
        print("判定：AASの見逃し率がRCより低い -> 「師匠を超えた」候補あり")
    elif aas_faster > total // 2:
        print("判定：AASがRCより早く検知 -> 「師匠を超えた」候補あり")
    else:
        print("判定：RCがAASより優位 -> 師匠はまだ超えられていない")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "e4_disciple_vs_master.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "E4",
            "model": adaptive_network.MODEL,
            "rc_detects": rc_detects,
            "aas_detects": aas_detects,
            "rc_miss_rate": rc_miss_rate,
            "aas_miss_rate": aas_miss_rate,
            "aas_faster_count": aas_faster,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n結果を {output_path} に保存しました")


if __name__ == "__main__":
    run_e4_experiment()
