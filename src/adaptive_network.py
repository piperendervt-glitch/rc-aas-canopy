"""
adaptive_network.py
可変構造ネットワーク：Node1 ⇄ Node2 ⇄ Node3 の双方向接続
flow_weightが動的に変化し、高い接続が優先される

RC+AAS Phase 1 (Seed) - ベースコードからの変更点：
  - 確率的ゆらぎ（ε ~ N(0, σ²)）をupdate_weightに追加
  - SIGMA / SIGMA_MAX をRCが管理する定数として定義
"""

import json
import random
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"
TIMEOUT = 60.0

# flow_weight 更新パラメータ
WEIGHT_INCREASE_RATE = 0.1   # 成功時: new = old + 0.1 * (1.0 - old)
WEIGHT_DECREASE_RATE = 0.7   # 失敗時: new = old * 0.7
INITIAL_WEIGHT = 0.5

# 確率的ゆらぎ（RCが管理）
SIGMA = 0.05        # ゆらぎの強さ（将来RCが調整）
SIGMA_MAX = 0.1     # 上限（RC設定）


def call_ollama(prompt: str, system: str = "") -> str:
    """Ollamaのモデルを呼び出す（標準ライブラリのみ使用）"""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        body = resp.read().decode("utf-8")
        result = json.loads(body)
        return result.get("response", "").strip()


@dataclass
class Connection:
    """ノード間の接続（flow_weightを持つ）"""
    from_node: int
    to_node: int
    flow_weight: float = INITIAL_WEIGHT
    history: List[float] = field(default_factory=list)

    def update_weight(self, success: bool, sigma: float = SIGMA):
        """結果に基づいてflow_weightを更新する（確率的ゆらぎ付き）"""
        self.history.append(self.flow_weight)

        if success:
            reward = WEIGHT_INCREASE_RATE * (1.0 - self.flow_weight)
        else:
            reward = -(self.flow_weight * (1.0 - WEIGHT_DECREASE_RATE))

        # ε ~ N(0, σ²)
        epsilon = random.gauss(0, min(sigma, SIGMA_MAX))

        # w_new = w_old + α×reward + ε
        self.flow_weight = self.flow_weight + reward + epsilon
        self.flow_weight = max(0.01, min(0.9, self.flow_weight))  # 上限0.9（第3条）

    def to_dict(self) -> dict:
        return {
            "from_node": self.from_node,
            "to_node": self.to_node,
            "flow_weight": self.flow_weight,
            "history": self.history,
        }


class AdaptiveNode:
    """可変ネットワークの各ノード"""

    def __init__(self, node_id: int):
        self.node_id = node_id

    def process_as_analyzer(self, world_rule: str, question: str, context: str = "") -> str:
        """ルール解析ノードとして動作"""
        system = "あなたはワールドルールを分析する専門家です。簡潔に回答してください。"
        prompt = (
            f"ワールドルール：{world_rule}\n"
            f"文章：「{question}」\n"
            f"{f'前のノードの分析: {context}' if context else ''}\n"
            f"このルールの核心と文章との関係を一文で分析してください。"
        )
        return call_ollama(prompt, system)

    def process_as_validator(self, rule_context: str, question: str, prev_analysis: str = "") -> str:
        """検証ノードとして動作"""
        system = "あなたは論理検証の専門家です。簡潔に回答してください。"
        prompt = (
            f"ルールの文脈：{rule_context}\n"
            f"文章：「{question}」\n"
            f"{f'前の分析: {prev_analysis}' if prev_analysis else ''}\n"
            f"文章はルールと整合していますか？一文で判断してください。"
        )
        return call_ollama(prompt, system)

    def process_as_judge(self, all_context: str, question: str) -> str:
        """最終判定ノードとして動作"""
        system = """必ず日本語で回答してください。
以下のフォーマットで答えてください：
判定：「矛盾しない」または「矛盾する」
確信度：0〜1の数値（1が最も確信が高い）

例：
判定：矛盾しない
確信度：0.85
"""
        prompt = (
            f"これまでの分析：{all_context}\n"
            f"元の文章：「{question}」\n"
            f"最終判定：上記フォーマットで答えてください。"
        )
        return call_ollama(prompt, system)


class AdaptiveNetwork:
    """
    可変構造ネットワーク
    Node1 ⇄ Node2 ⇄ Node3 の双方向接続
    flow_weightが高い接続が優先される
    """

    def __init__(self):
        self.nodes = {
            1: AdaptiveNode(1),
            2: AdaptiveNode(2),
            3: AdaptiveNode(3),
        }

        # 双方向接続の初期化
        self.connections: Dict[Tuple[int, int], Connection] = {
            (1, 2): Connection(1, 2),
            (2, 1): Connection(2, 1),
            (2, 3): Connection(2, 3),
            (3, 2): Connection(3, 2),
            (1, 3): Connection(1, 3),  # スキップ接続
            (3, 1): Connection(3, 1),  # スキップ接続（逆方向）
        }

        # 実験ログ
        self.weight_log: List[dict] = []

    def _get_active_path(self) -> List[Tuple[int, int]]:
        """現在のflow_weightに基づいて最適な処理パスを選択する"""
        path_full = [(1, 2), (2, 3)]
        path_short = [(1, 3)]

        weight_full = min(
            self.connections[(1, 2)].flow_weight,
            self.connections[(2, 3)].flow_weight,
        )
        weight_short = self.connections[(1, 3)].flow_weight

        if weight_full >= weight_short:
            return path_full
        else:
            return path_short

    def _should_use_feedback(self) -> bool:
        """逆方向フィードバックを使うか判定"""
        feedback_weight = self.connections[(3, 2)].flow_weight
        return feedback_weight > 0.5

    def predict(self, world_rule: str, question: str) -> dict:
        """flow_weightに基づく動的パスで予測する"""
        active_path = self._get_active_path()
        use_feedback = self._should_use_feedback()
        node_outputs = {}
        node_results = []

        try:
            if active_path == [(1, 3)]:
                output1 = self.nodes[1].process_as_analyzer(world_rule, question)
                node_outputs[1] = output1
                node_results.append({"node": 1, "output": output1, "role": "analyzer"})

                final_output = self.nodes[3].process_as_judge(output1, question)
                node_outputs[3] = final_output
                node_results.append({"node": 3, "output": final_output, "role": "judge"})

            else:
                output1 = self.nodes[1].process_as_analyzer(world_rule, question)
                node_outputs[1] = output1
                node_results.append({"node": 1, "output": output1, "role": "analyzer"})

                output2 = self.nodes[2].process_as_validator(output1, question)
                node_outputs[2] = output2
                node_results.append({"node": 2, "output": output2, "role": "validator"})

                if use_feedback:
                    refined_output1 = self.nodes[1].process_as_analyzer(
                        world_rule, question, context=output2
                    )
                    node_outputs[1] = refined_output1
                    node_results.append({"node": 1, "output": refined_output1, "role": "re-analyzer"})
                    all_context = f"{refined_output1} | {output2}"
                else:
                    all_context = f"{output1} | {output2}"

                final_output = self.nodes[3].process_as_judge(all_context, question)
                node_outputs[3] = final_output
                node_results.append({"node": 3, "output": final_output, "role": "judge"})

        except Exception as e:
            final_output = f"ERROR: {e}"
            node_results.append({"node": "error", "output": final_output, "role": "error"})

        prediction = self._parse_prediction(final_output)
        confidence = self._parse_confidence(final_output)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "raw_output": final_output,
            "path_used": active_path,
            "node_results": node_results,
            "used_feedback": use_feedback,
            "flow_weights": self.get_weights_snapshot(),
        }

    def _edge_to_arm_id(self, edge: Tuple[int, int]) -> str:
        """エッジタプルをarm_id文字列に変換"""
        return f"{edge[0]}->{edge[1]}"

    def update_weights(self, success: bool, path_used: List[Tuple[int, int]], used_feedback: bool,
                       sigma: float = SIGMA, sealed_paths: set = None):
        """予測結果に基づいてflow_weightを更新する（σはRCが渡す）"""
        sealed = sealed_paths or set()
        for edge in path_used:
            if edge in self.connections and self._edge_to_arm_id(edge) not in sealed:
                self.connections[edge].update_weight(success, sigma=sigma)

        if used_feedback:
            if self._edge_to_arm_id((3, 2)) not in sealed:
                self.connections[(3, 2)].update_weight(success, sigma=sigma)
            if self._edge_to_arm_id((2, 1)) not in sealed:
                self.connections[(2, 1)].update_weight(success, sigma=sigma)

        for key, conn in self.connections.items():
            if key not in path_used and self._edge_to_arm_id(key) not in sealed:
                if not (used_feedback and key in [(3, 2), (2, 1)]):
                    conn.flow_weight = conn.flow_weight * 0.99  # Phase 1確定値（案3はPhase 2へ）

        self.weight_log.append(self.get_weights_snapshot())

    def decay_weights(self, decay_rate: float = 0.995, exclude_path: list = None, sealed_paths: set = None):
        """
        案7（時間減衰）：使用パスのみにdecayを適用する。

        設計原則（確定）：
          - active_path（使用中）にのみdecayを適用（過集中を防ぐ）
          - 非使用パスはupdate内0.99に任せる（二重減衰禁止）

        decay_rate: 使用パスの減衰率（デフォルト0.995）
        exclude_path: decay対象外のエッジ（通常は非使用パス＝Noneで全パス対象）
          ※呼び出し側がactive_pathを渡すこと
        """
        sealed = sealed_paths or set()
        exclude = set(tuple(e) for e in exclude_path) if exclude_path else set()
        for key, conn in self.connections.items():
            if self._edge_to_arm_id(key) in sealed:
                continue  # 封印済みパスはdecayスキップ
            if key not in exclude:
                pass  # 非使用パスはupdate内0.99に任せる（二重減衰禁止）
            else:
                # 使用パスにのみdecayを適用
                conn.flow_weight = max(0.01, conn.flow_weight * decay_rate)

    def get_weights_snapshot(self) -> dict:
        return {
            f"{k[0]}->{k[1]}": round(v.flow_weight, 4)
            for k, v in self.connections.items()
        }

    def get_weight_history(self) -> dict:
        return {
            f"{k[0]}->{k[1]}": v.history
            for k, v in self.connections.items()
        }

    def _parse_confidence(self, raw_output: str) -> float:
        """確信度を抽出する。取得できない場合は0.5（判断保留）を返す。"""
        import re
        match = re.search(r'確信度[：:]\s*([0-9.]+)', raw_output)
        if match:
            try:
                val = float(match.group(1))
                return max(0.0, min(1.0, val))
            except ValueError:
                pass
        return 0.5

    def _parse_prediction(self, output: str) -> bool:
        if "矛盾しない" in output:
            return True
        if "矛盾する" in output:
            return False
        return False


if __name__ == "__main__":
    network = AdaptiveNetwork()
    result = network.predict(
        world_rule="この世界では空は緑色である",
        question="空を見上げると緑色が広がっていた",
    )
    print(f"予測: {'矛盾しない' if result['prediction'] else '矛盾する'}")
    print(f"パス: {result['path_used']}")
    print(f"flow_weights: {result['flow_weights']}")
    network.update_weights(True, result["path_used"], result["used_feedback"])
    print(f"更新後weights: {network.get_weights_snapshot()}")
