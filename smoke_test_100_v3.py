# smoke_test_100_v3.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from task_generator import generate_tasks
from adaptive_network import AdaptiveNetwork
from rc import RC, DECAY_RATE

tasks = generate_tasks()  # 全100問
rc = RC()
network = AdaptiveNetwork()
correct = 0

for i, task in enumerate(tasks):
    output = network.predict(task.world_rule, task.question)
    is_correct = (output["prediction"] == task.label)
    if is_correct:
        correct += 1

    network.update_weights(
        success=is_correct,
        path_used=output["path_used"],
        used_feedback=output["used_feedback"],
        sigma=rc.get_sigma(),
    )
    network.decay_weights(decay_rate=DECAY_RATE, exclude_path=output["path_used"])

    alerts = rc.monitor(network.get_weights_snapshot(), {"overall": round(correct / (i + 1), 4)})
    status = "OK" if is_correct else "NG"
    w = network.get_weights_snapshot()
    print(f"[{i+1:3d}] {status} | 1->2={w.get('1->2',0):.3f} 2->3={w.get('2->3',0):.3f} 1->3={w.get('1->3',0):.3f} | alerts:{len(alerts)}")

    if (i + 1) % 10 == 0:
        print(f"\n--- {i+1}問完了 | 正解率: {correct}/{i+1} ({correct/(i+1):.0%}) ---")
        print(f"RC: {rc.dump_state()}\n")

print(f"\n最終正解率: {correct}/100 ({correct/100:.0%})")
print(f"RC最終状態: {rc.dump_state()}")
print(f"最終weights: {network.get_weights_snapshot()}")
