"""
fixed_network.py
固定構造ネットワーク：Node1 → Node2 → Node3 の固定順序処理
各ノードはqwen2.5:3b（Ollama）を呼び出す
"""

import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"
TIMEOUT = 60.0


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
class NodeResult:
    node_id: int
    input_text: str
    output_text: str
    success: bool


class Node1:
    """
    Node1: 文章とルールを受け取り、ルールの要点を整理する
    役割: 前処理・ルール解析
    """
    node_id = 1

    def process(self, world_rule: str, question: str) -> NodeResult:
        system = "あなたはワールドルールを分析する専門家です。簡潔に回答してください。"
        prompt = (
            f"ワールドルール：{world_rule}\n"
            f"このルールの核心を一文で要約してください。"
        )
        try:
            output = call_ollama(prompt, system)
            return NodeResult(
                node_id=self.node_id,
                input_text=f"{world_rule} | {question}",
                output_text=output,
                success=True,
            )
        except Exception as e:
            return NodeResult(
                node_id=self.node_id,
                input_text=f"{world_rule} | {question}",
                output_text=f"ERROR: {e}",
                success=False,
            )


class Node2:
    """
    Node2: Node1の出力を受け取り、文章の意味を解析する
    役割: 文章解析
    """
    node_id = 2

    def process(self, rule_summary: str, question: str) -> NodeResult:
        system = "あなたは文章分析の専門家です。簡潔に回答してください。"
        prompt = (
            f"ルールの要点：{rule_summary}\n"
            f"文章：「{question}」\n"
            f"この文章が示している状態を一文で説明してください。"
        )
        try:
            output = call_ollama(prompt, system)
            return NodeResult(
                node_id=self.node_id,
                input_text=f"{rule_summary} | {question}",
                output_text=output,
                success=True,
            )
        except Exception as e:
            return NodeResult(
                node_id=self.node_id,
                input_text=f"{rule_summary} | {question}",
                output_text=f"ERROR: {e}",
                success=False,
            )


class Node3:
    """
    Node3: Node1・Node2の出力を受け取り、矛盾の有無を判定する
    役割: 最終判定
    """
    node_id = 3

    def process(self, rule_summary: str, sentence_analysis: str, question: str) -> NodeResult:
        system = "あなたは論理的な判定者です。「矛盾しない」または「矛盾する」とだけ答えてください。"
        prompt = (
            f"ルールの要点：{rule_summary}\n"
            f"文章の内容：{sentence_analysis}\n"
            f"元の文章：「{question}」\n"
            f"この文章はルールと矛盾しますか？\n"
            f"「矛盾しない」または「矛盾する」とだけ答えてください。"
        )
        try:
            output = call_ollama(prompt, system)
            return NodeResult(
                node_id=self.node_id,
                input_text=f"{rule_summary} | {sentence_analysis}",
                output_text=output,
                success=True,
            )
        except Exception as e:
            return NodeResult(
                node_id=self.node_id,
                input_text=f"{rule_summary} | {sentence_analysis}",
                output_text=f"ERROR: {e}",
                success=False,
            )


class FixedNetwork:
    """
    固定構造ネットワーク
    Node1 → Node2 → Node3 の固定順序で処理する
    接続・重みは変化しない
    """

    def __init__(self):
        self.node1 = Node1()
        self.node2 = Node2()
        self.node3 = Node3()

    def predict(self, world_rule: str, question: str) -> dict:
        """
        3ノードを固定順序で実行し、矛盾の有無を予測する
        Returns: {
            'prediction': bool,   # True=矛盾しない / False=矛盾する
            'raw_output': str,    # Node3の生出力
            'node_results': list, # 各ノードの結果
        }
        """
        # Node1: ルール解析
        result1 = self.node1.process(world_rule, question)

        # Node2: 文章解析（Node1が失敗した場合は元のルールを使用）
        rule_summary = result1.output_text if result1.success else world_rule
        result2 = self.node2.process(rule_summary, question)

        # Node3: 最終判定
        sentence_analysis = result2.output_text if result2.success else question
        result3 = self.node3.process(rule_summary, sentence_analysis, question)

        # 出力をパース
        prediction = self._parse_prediction(result3.output_text)

        return {
            "prediction": prediction,
            "raw_output": result3.output_text,
            "node_results": [
                {"node": 1, "output": result1.output_text, "success": result1.success},
                {"node": 2, "output": result2.output_text, "success": result2.success},
                {"node": 3, "output": result3.output_text, "success": result3.success},
            ],
        }

    def _parse_prediction(self, output: str) -> bool:
        """出力テキストから予測値を取得する"""
        output_lower = output.lower()
        if "矛盾しない" in output:
            return True
        if "矛盾する" in output:
            return False
        # 不明瞭な回答はデフォルトでFalse
        return False


if __name__ == "__main__":
    network = FixedNetwork()
    result = network.predict(
        world_rule="この世界では空は緑色である",
        question="空を見上げると緑色が広がっていた",
    )
    print(f"予測: {'矛盾しない' if result['prediction'] else '矛盾する'}")
    print(f"出力: {result['raw_output']}")
