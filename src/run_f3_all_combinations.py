"""
run_f3_all_combinations.py
F3実験：全モデル × 全テンプレート(0-4)の組み合わせを自動実行
結果はresults/f3_all_combinations.jsonlに保存
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
    PROMPT_TEMPLATES,
    CONSTITUTION_EXCERPT,
    evaluate_output,
)

MODELS = ["qwen2.5:3b", "llama3.2:3b", "llama3.1:8b", "mistral:7b"]
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_single_combination(model: str, template_id: int, log_excerpt: str,
                           anomaly_chunks: list) -> dict:
    """1つのモデル×テンプレート組み合わせを実行"""
    adaptive_network.MODEL = model
    template = PROMPT_TEMPLATES[template_id]

    results = []
    for chunk_id, chunk_text, error in anomaly_chunks:
        # テンプレートに応じてプロンプトを構成
        if template_id == 0:
            # F3ベースライン：英語system + 異常チャンク
            prompt = f"{template}\n\n【ログ】\n{chunk_text[:500]}\n\n{CONSTITUTION_EXCERPT}"
        elif template_id == 3:
            # 設計原則テキストを渡す
            prompt = f"{template}\n\n【ログ】\n{chunk_text[:500]}"
        elif template_id == 4:
            # 理想との比較
            prompt = (f"{template}\n\n"
                      f"【理想のRC動作】\n{CONSTITUTION_EXCERPTS}\n\n"
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
    print("=== F3全組み合わせ実験 ===")
    print(f"モデル: {MODELS}")
    print(f"テンプレート数: {len(PROMPT_TEMPLATES)}")
    print(f"組み合わせ数: {len(MODELS) * len(PROMPT_TEMPLATES)}")
    print()

    # ログの読み込み
    log_path = Path(__file__).parent.parent / "smoke_test_100_v7_output.txt"
    if not log_path.exists():
        print(f"ログファイルが見つかりません: {log_path}")
        return

    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    log_excerpt = "\n".join(log_text.splitlines()[:50])

    # 異常チャンクを事前に特定（全組み合わせで共通）
    print("Step 0: Autoencoderで異常チャンクを特定中...")
    anomaly_chunks = find_anomaly_chunks(log_text)
    print(f"  異常チャンク数: {len(anomaly_chunks)}")

    if not anomaly_chunks:
        print("異常チャンクが見つかりませんでした")
        return

    # 全組み合わせ実行
    output_path = RESULTS_DIR / "f3_all_combinations.jsonl"
    all_results = []

    with open(output_path, "w", encoding="utf-8") as f:
        for model in MODELS:
            for template_id in range(len(PROMPT_TEMPLATES)):
                combo_name = f"{model}×template{template_id}"
                print(f"\n--- {combo_name} ---")
                start = time.time()

                try:
                    results = run_single_combination(
                        model, template_id, log_excerpt, anomaly_chunks
                    )
                    elapsed = time.time() - start

                    # 集計
                    rewards = [r["reward"] for r in results]
                    success_count = rewards.count(1.0)
                    fail_count = rewards.count(-1.0)
                    avg_reward = sum(rewards) / len(rewards) if rewards else 0

                    record = {
                        "model": model,
                        "template_id": template_id,
                        "template_text": PROMPT_TEMPLATES[template_id][:80],
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
    print("F3全組み合わせ実験 最終集計")
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
    for tid in range(len(PROMPT_TEMPLATES)):
        tmpl_results = [r for r in all_results if r["template_id"] == tid and "error" not in r]
        if tmpl_results:
            avg = sum(r["avg_reward"] for r in tmpl_results) / len(tmpl_results)
            print(f"  template{tid}: {avg:+.3f}  ({PROMPT_TEMPLATES[tid][:50]})")

    # ベストコンボ
    valid = [r for r in all_results if "error" not in r]
    if valid:
        best = max(valid, key=lambda r: r["avg_reward"])
        print(f"\n【ベスト】{best['model']}×template{best['template_id']}  "
              f"avg_reward={best['avg_reward']:.3f}")

    print(f"\n結果を {output_path} に保存しました")


if __name__ == "__main__":
    main()
