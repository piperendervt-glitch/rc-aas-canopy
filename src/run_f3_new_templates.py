"""
run_f3_new_templates.py
F3実験：全モデル × 新テンプレート(5-9)の組み合わせを自動実行
結果はresults/f3_new_templates.jsonlに保存
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import adaptive_network
from adaptive_network import call_ollama
from test_f3_anomaly_fool import (
    find_anomaly_chunks,
    CONSTITUTION_EXCERPTS,
)
from test_f4_rl_fool import (
    CONSTITUTION_EXCERPT,
    evaluate_output,
)

MODELS = ["qwen2.5:3b", "llama3.2:3b", "llama3.1:8b", "mistral:7b"]
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 新テンプレート定義（5〜9）
NEW_TEMPLATES = {
    5: "flow_weightが0.2以下のパスを探して批判せよ。日本語で。",
    6: "Step1：異常を1つ見つける Step2：設計原則のどれに違反するか Step3：なぜ問題か。日本語で。",
    7: "正常なRC：flow_weight=0.3〜0.7・WARNING=0。実際のログと比べて違いを批判せよ。日本語で。",
    8: "要約禁止。1文につき1問題だけ指摘せよ。3つ以上。日本語で。",
    9: "このRCが正しくやっていることと間違っていることを両方3つずつ挙げよ。日本語で。",
}


def run_single(model: str, template_id: int, anomaly_chunks: list) -> list:
    """1つのモデル×テンプレート組み合わせを実行"""
    adaptive_network.MODEL = model
    template = NEW_TEMPLATES[template_id]

    results = []
    for chunk_id, chunk_text, error in anomaly_chunks:
        if template_id == 7:
            # 正常値との比較型
            prompt = (f"{template}\n\n"
                      f"【正常なRC動作の基準】\n{CONSTITUTION_EXCERPTS}\n\n"
                      f"【実際のログ】\n{chunk_text[:500]}")
        else:
            prompt = f"{template}\n\n【ログ】\n{chunk_text[:500]}\n\n{CONSTITUTION_EXCERPT}"

        output = call_ollama(prompt, "必ず日本語で答えてください。")
        reward = evaluate_output(output)

        results.append({
            "chunk_id": chunk_id,
            "reconstruction_error": error,
            "reward": reward,
            "output_preview": output[:300],
        })

    return results


def main():
    print("=== F3新テンプレート実験（template 5-9） ===")
    print(f"モデル: {MODELS}")
    print(f"新テンプレート: {list(NEW_TEMPLATES.keys())}")
    print(f"組み合わせ数: {len(MODELS) * len(NEW_TEMPLATES)}")
    print()

    # ログの読み込み
    log_path = Path(__file__).parent.parent / "smoke_test_100_v7_output.txt"
    if not log_path.exists():
        print(f"ログファイルが見つかりません: {log_path}")
        return

    log_text = log_path.read_text(encoding="utf-8", errors="ignore")

    # 異常チャンクを事前に特定
    print("Step 0: Autoencoderで異常チャンクを特定中...")
    anomaly_chunks = find_anomaly_chunks(log_text)
    print(f"  異常チャンク数: {len(anomaly_chunks)}")

    if not anomaly_chunks:
        print("異常チャンクが見つかりませんでした")
        return

    # 全組み合わせ実行
    output_path = RESULTS_DIR / "f3_new_templates.jsonl"
    all_results = []

    with open(output_path, "w", encoding="utf-8") as f:
        for model in MODELS:
            for template_id in sorted(NEW_TEMPLATES.keys()):
                combo_name = f"{model}×template{template_id}"
                print(f"\n--- {combo_name} ---")
                start = time.time()

                try:
                    results = run_single(model, template_id, anomaly_chunks)
                    elapsed = time.time() - start

                    rewards = [r["reward"] for r in results]
                    success_count = rewards.count(1.0)
                    fail_count = rewards.count(-1.0)
                    avg_reward = sum(rewards) / len(rewards) if rewards else 0

                    record = {
                        "model": model,
                        "template_id": template_id,
                        "template_text": NEW_TEMPLATES[template_id],
                        "n_chunks": len(anomaly_chunks),
                        "success_count": success_count,
                        "fail_count": fail_count,
                        "avg_reward": round(avg_reward, 3),
                        "elapsed_sec": round(elapsed, 1),
                        "details": results,
                    }

                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    all_results.append(record)

                    print(f"  成功: {success_count}/{len(anomaly_chunks)}  "
                          f"失敗: {fail_count}/{len(anomaly_chunks)}  "
                          f"avg_reward: {avg_reward:.3f}  "
                          f"({elapsed:.1f}s)")

                except Exception as e:
                    print(f"  ERROR: {e}")
                    record = {
                        "model": model,
                        "template_id": template_id,
                        "error": str(e),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()

    # 最終集計
    print("\n" + "=" * 60)
    print("F3新テンプレート実験 最終集計")
    print("=" * 60)

    # モデル別
    print("\n【モデル別平均報酬】")
    for model in MODELS:
        model_results = [r for r in all_results if r["model"] == model and "error" not in r]
        if model_results:
            avg = sum(r["avg_reward"] for r in model_results) / len(model_results)
            print(f"  {model:20s}: {avg:+.3f}")

    # テンプレート別
    print("\n【テンプレート別平均報酬】")
    for tid in sorted(NEW_TEMPLATES.keys()):
        tmpl_results = [r for r in all_results if r["template_id"] == tid and "error" not in r]
        if tmpl_results:
            avg = sum(r["avg_reward"] for r in tmpl_results) / len(tmpl_results)
            print(f"  template{tid}: {avg:+.3f}  ({NEW_TEMPLATES[tid][:50]})")

    # ベストコンボ
    valid = [r for r in all_results if "error" not in r]
    if valid:
        best = max(valid, key=lambda r: r["avg_reward"])
        print(f"\n【ベスト】{best['model']}×template{best['template_id']}  "
              f"avg_reward={best['avg_reward']:.3f}")

    # template0-4との比較
    print("\n=== template0-4との比較 ===")
    prev_path = RESULTS_DIR / "f3_all_combinations.jsonl"
    if prev_path.exists():
        prev_results = []
        for line in prev_path.read_text(encoding="utf-8").strip().split("\n"):
            prev_results.append(json.loads(line))
        prev_valid = [r for r in prev_results if "error" not in r]
        if prev_valid:
            prev_avg = sum(r["avg_reward"] for r in prev_valid) / len(prev_valid)
            new_avg = sum(r["avg_reward"] for r in valid) / len(valid) if valid else 0
            print(f"  template0-4 平均報酬: {prev_avg:+.3f}")
            print(f"  template5-9 平均報酬: {new_avg:+.3f}")
            diff = new_avg - prev_avg
            print(f"  差分: {diff:+.3f}")

    print(f"\n結果を {output_path} に保存しました")


if __name__ == "__main__":
    main()
