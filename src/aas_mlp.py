"""
aas_mlp.py
AAS層のMLP実装
LLMが抽出した特徴量（5次元）を受け取って判定する
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "results" / "aas_mlp.pt"


class AAS_MLP(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, features: list[float]) -> tuple[bool, float]:
        x = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            score = self.forward(x).item()
        return score > 0.5, score

    def save(self):
        torch.save(self.state_dict(), MODEL_PATH)

    def load(self):
        if MODEL_PATH.exists():
            self.load_state_dict(torch.load(MODEL_PATH))


if __name__ == "__main__":
    mlp = AAS_MLP()
    test_features = [1.0, 0.95, 1.0, 0.0, 0.9]
    prediction, score = mlp.predict(test_features)
    print(f"入力: {test_features}")
    print(f"予測: {'矛盾しない' if prediction else '矛盾する'} (score={score:.4f})")
