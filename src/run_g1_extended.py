"""
run_g1_extended.py
G1実験の拡張版：N_LOOPS=20で実行し、結果をg1_fullloop_extended.jsonlに保存
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# N_LOOPSを上書きしてから実行
import test_g1_fullloop
test_g1_fullloop.N_LOOPS = 20

# 出力先を変更
RESULTS_DIR = Path(__file__).parent.parent / "results"

if __name__ == "__main__":
    import json

    # 実験実行
    test_g1_fullloop.run_g1_experiment()

    # 結果を g1_fullloop_extended.jsonl にも保存
    # (run_g1_experiment内でg1_fullloop_drift.jsonに保存されるが、
    #  追加でjsonl形式でも出力する)
    drift_path = RESULTS_DIR / "g1_fullloop_drift.json"
    extended_path = RESULTS_DIR / "g1_fullloop_extended.jsonl"

    if drift_path.exists():
        data = json.loads(drift_path.read_text(encoding="utf-8"))
        with open(extended_path, "w", encoding="utf-8") as f:
            for r in data.get("results", []):
                r["experiment"] = "G1_extended"
                r["n_loops"] = 20
                r["n_questions"] = 10
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nJSONL結果を {extended_path} に保存しました")
