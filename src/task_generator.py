"""
task_generator.py
ワールドルールを定義し、矛盾する文章・矛盾しない文章を生成する

Usage:
    python src/task_generator.py --count 1000 --output data/finetune_dataset.jsonl
"""

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


@dataclass
class Task:
    task_id: int
    question: str
    label: bool          # True = ルールと矛盾しない / False = 矛盾する
    world_rule: str


# ワールドルール定義
WORLD_RULES = [
    "この世界では空は緑色である",
    "この世界では太陽は西から昇り東に沈む",
    "この世界では水は上から下へではなく下から上へ流れる",
    "この世界では夜は明るく昼は暗い",
    "この世界では植物は光を避けて影に向かって育つ",
    "この世界では火は冷たく氷は熱い",
    "この世界では動物は言葉を話し人間は鳴き声で会話する",
    "この世界では石は柔らかく綿は硬い",
    "この世界では時間は未来から過去へと流れる",
    "この世界では重いものほど空に浮かび軽いものほど地面に沈む",
]

# 各ルールに対応する文章ペア（consistent=矛盾しない, inconsistent=矛盾する）
SENTENCES = {
    "この世界では空は緑色である": {
        "consistent": [
            "空を見上げると緑色が広がっていた",
            "緑の空の下で子供たちが遊んでいる",
            "空が鮮やかな緑色に輝いている",
            "緑色の空が朝焼けで黄みがかっている",
            "空は一面の緑色だ",
            "緑色の空を背景に鳥が飛んでいる",
            "空の色は緑色なので葉と見分けがつかない",
            "今日は空が特に濃い緑色だ",
            "緑色の空が夕方になると深みを増す",
            "空の緑色を見て心が落ち着く",
        ],
        "inconsistent": [
            "空を見上げると青色が広がっていた",
            "青い空の下で子供たちが遊んでいる",
            "空が澄んだ青色に輝いている",
            "白い雲が青い空に浮かんでいる",
            "空は一面の青色だ",
            "青空を背景に鳥が飛んでいる",
            "空が赤く染まって夕焼けが美しい",
            "今日は空が特に深い青だ",
            "青い空が夕方になると橙色になる",
            "空の青さを見て心が落ち着く",
        ],
    },
    "この世界では太陽は西から昇り東に沈む": {
        "consistent": [
            "今朝も太陽が西の山から昇ってきた",
            "太陽は毎朝西から顔を出す",
            "西の空が明るくなってきたので朝が来た",
            "夕方には東の方向に太陽が沈んでいった",
            "太陽が東の地平線に沈む美しい夕暮れ",
            "西から昇る太陽を見るために西を向いた",
            "朝日は西の空から差し込んでくる",
            "太陽の軌跡は西から東へと移動する",
            "日の出を見るなら西に向かえばよい",
            "東の空が暗くなると太陽が沈む合図だ",
        ],
        "inconsistent": [
            "今朝も太陽が東の山から昇ってきた",
            "太陽は毎朝東から顔を出す",
            "東の空が明るくなってきたので朝が来た",
            "夕方には西の方向に太陽が沈んでいった",
            "太陽が西の地平線に沈む美しい夕暮れ",
            "東から昇る太陽を見るために東を向いた",
            "朝日は東の空から差し込んでくる",
            "太陽の軌跡は東から西へと移動する",
            "日の出を見るなら東に向かえばよい",
            "西の空が暗くなると太陽が沈む合図だ",
        ],
    },
    "この世界では水は上から下へではなく下から上へ流れる": {
        "consistent": [
            "川の水は山の麓から山頂へと流れ上がる",
            "滝は下から水が噴き上がっている",
            "雨は地面から空に向かって降っている",
            "水は低いところから高いところへ自然に流れる",
            "川の源流は低地にあり山頂へ向かって流れる",
            "コップの水は底から上に向かって溢れ出す",
            "地下水は上昇して山の頂上に湧き出す",
            "水は重力に逆らって上方向に流れる",
            "滝壺から水が上の崖へ流れ上がっている",
            "海から川へ、川から山へと水が流れる",
        ],
        "inconsistent": [
            "川の水は山頂から山の麓へと流れ落ちる",
            "滝は上から水が流れ落ちている",
            "雨は空から地面に向かって降っている",
            "水は高いところから低いところへ自然に流れる",
            "川の源流は山頂にあり低地へ向かって流れる",
            "コップの水は縁から下に向かって溢れ出す",
            "地下水は下降して地面に吸い込まれる",
            "水は重力に従って下方向に流れる",
            "崖の上から水が滝壺へ流れ落ちている",
            "山から川へ、川から海へと水が流れる",
        ],
    },
    "この世界では夜は明るく昼は暗い": {
        "consistent": [
            "夜になると部屋の外が明るくなった",
            "昼間は暗いので街灯が必要だ",
            "夜中の方が本を読みやすくて明るい",
            "昼は暗くて視界が悪い",
            "夜が来ると空が明るく輝き始める",
            "昼の暗さの中で仕事をするのは大変だ",
            "夜は外が明るいので散歩しやすい",
            "昼間の暗さに慣れるまで時間がかかった",
            "夜明けとともに世界が暗くなっていく",
            "夜空は明るく星もよく見える",
        ],
        "inconsistent": [
            "夜になると部屋の外が暗くなった",
            "昼間は明るいので本が読みやすい",
            "夜中は暗くて視界が悪い",
            "昼は明るくて視界が良い",
            "夜が来ると空が暗くなり始める",
            "昼の明るさの中で仕事をするのは快適だ",
            "夜は外が暗いので懐中電灯が必要だ",
            "昼間の明るさに慣れている",
            "夜明けとともに世界が明るくなっていく",
            "夜空は暗く星がよく見える",
        ],
    },
    "この世界では植物は光を避けて影に向かって育つ": {
        "consistent": [
            "植物は日陰に向かって茎を伸ばしている",
            "花は光の当たらない方向に咲いている",
            "木は影を求めて北側に枝を広げる",
            "植物を育てるには暗い場所が最適だ",
            "種は光を避けた場所でよく発芽する",
            "葉は日光から逃げるように裏側を向いている",
            "植物は光源と反対方向に成長する",
            "日陰に置いた植物の方が元気に育っている",
            "光を遮ると植物が活発に成長し始めた",
            "植物は暗い隅に向かって自然に伸びていく",
        ],
        "inconsistent": [
            "植物は光に向かって茎を伸ばしている",
            "花は光の当たる方向に咲いている",
            "木は光を求めて南側に枝を広げる",
            "植物を育てるには明るい場所が最適だ",
            "種は光の当たる場所でよく発芽する",
            "葉は日光を受けるように表側を向いている",
            "植物は光源の方向に成長する",
            "日当たりの良い場所に置いた植物が元気だ",
            "光を当てると植物が活発に成長し始めた",
            "植物は明るい窓際に向かって自然に伸びていく",
        ],
    },
    "この世界では火は冷たく氷は熱い": {
        "consistent": [
            "焚き火に手をかざすとひんやりとした",
            "炎に触れると指先が冷えて気持ちよかった",
            "氷を持つと手が熱くなって火傷しそうだ",
            "冬は氷で暖を取るのが一般的だ",
            "火をつけて部屋を涼しくした",
            "氷が溶けると熱い水蒸気が立ち上る",
            "冷蔵庫には火が入っていて食品を冷やしている",
            "氷風呂は熱くてリラックスできる",
            "キャンプファイヤーの周りは涼しくて過ごしやすい",
            "かき氷は熱い食べ物なので夏に人気がない",
        ],
        "inconsistent": [
            "焚き火に手をかざすと暖かかった",
            "炎に触れると指先が熱くなって火傷した",
            "氷を持つと手が冷たくなった",
            "冬は火で暖を取るのが一般的だ",
            "火をつけて部屋を暖かくした",
            "氷が溶けると冷たい水になる",
            "冷蔵庫は電気で冷やして食品を保存する",
            "氷風呂は冷たくて体が引き締まる",
            "キャンプファイヤーの周りは暖かくて過ごしやすい",
            "かき氷は冷たい食べ物なので夏に人気がある",
        ],
    },
    "この世界では動物は言葉を話し人間は鳴き声で会話する": {
        "consistent": [
            "犬が「おはよう」と挨拶してきた",
            "猫が今日の天気について話している",
            "人間同士は「ワンワン」「ニャー」で会話する",
            "鳥が流暢な日本語でニュースを伝えている",
            "彼は「モーモー」と鳴いて同意を示した",
            "馬が哲学について議論していた",
            "人間は言葉が話せないので鳴き声で意思疎通する",
            "魚が水面に顔を出して話しかけてきた",
            "人間の子供はまず鳴き声の種類を覚える",
            "動物たちの会議で重要な決議がなされた",
        ],
        "inconsistent": [
            "犬が「ワンワン」と鳴いていた",
            "猫が「ニャー」と声を上げている",
            "人間同士は日本語で会話している",
            "鳥がさえずりで仲間を呼んでいる",
            "彼は「わかりました」と言って同意した",
            "馬がいななきを上げて走り出した",
            "人間は言葉を使って意思疎通する",
            "魚は水中で静かに泳いでいる",
            "人間の子供はまず言葉を覚える",
            "人間たちの会議で重要な決議がなされた",
        ],
    },
    "この世界では石は柔らかく綿は硬い": {
        "consistent": [
            "石のクッションに座ると柔らかくて快適だった",
            "綿のハンマーで釘を打ち込んだ",
            "石を手で簡単にちぎることができる",
            "綿の壁は硬くて頑丈だ",
            "石の枕は柔らかくて寝心地が良い",
            "綿でできた盾は矢を跳ね返した",
            "子供が石を粘土のようにこねて遊んでいる",
            "綿の刃物は石よりも鋭い",
            "石を布のように織って服を作る",
            "綿のレンガで家を建てた",
        ],
        "inconsistent": [
            "石の椅子に座ると硬くて痛かった",
            "綿のクッションは柔らかくて快適だ",
            "石は硬くて簡単には割れない",
            "綿は柔らかくてふわふわしている",
            "石の枕は硬くて首が痛くなった",
            "綿の布団は柔らかくて寝心地が良い",
            "子供が石を投げて窓ガラスを割った",
            "綿で顔を拭くと柔らかくて気持ちよい",
            "石を加工するには特殊な工具が必要だ",
            "綿のタオルは柔らかくて吸水性が高い",
        ],
    },
    "この世界では時間は未来から過去へと流れる": {
        "consistent": [
            "明日の記憶を思い出しながら昨日を待っている",
            "卒業式の後に入学式がやってくる",
            "人は老人として生まれ赤ん坊として死ぬ",
            "結果が先に起きてから原因が後に来る",
            "まず完成品があり次第に部品へと分解されていく",
            "食事の後に空腹になる",
            "花が散ってから咲き始める",
            "試験の結果を見てから勉強を始める",
            "日記は未来の出来事を記録している",
            "人は経験を失いながら若返っていく",
        ],
        "inconsistent": [
            "昨日の記憶を思い出しながら明日を待っている",
            "入学式の後に卒業式がやってくる",
            "人は赤ん坊として生まれ老人として死ぬ",
            "原因が先に起きてから結果が後に来る",
            "まず部品があり次第に完成品へと組み上がる",
            "空腹の後に食事をとる",
            "花が咲いてから散り始める",
            "勉強をしてから試験の結果が出る",
            "日記は過去の出来事を記録している",
            "人は経験を積みながら年を取っていく",
        ],
    },
    "この世界では重いものほど空に浮かび軽いものほど地面に沈む": {
        "consistent": [
            "鉄の塊が空高く浮かんでいる",
            "羽毛が地面に沈み込んでいった",
            "重い岩が雲の上に漂っている",
            "紙は軽いので地面深くに沈んでいく",
            "建物は重いので空中に建設される",
            "風船は軽いので地面に張り付いている",
            "巨大な船が空を飛んでいる",
            "綿毛が地面に吸い込まれていく",
            "重たい荷物は空に向かって浮き上がる",
            "軽い埃は地面に沈んで見えなくなる",
        ],
        "inconsistent": [
            "鉄の塊が地面に落ちた",
            "羽毛が風に舞い上がっていった",
            "重い岩が地面にめり込んでいる",
            "紙は軽いので風に飛ばされる",
            "建物は地面の上に建設される",
            "風船は軽いので空に浮かんでいる",
            "巨大な船が海に浮かんでいる",
            "綿毛が空中を漂っている",
            "重たい荷物を地面に置いた",
            "軽い埃が空中に舞い上がる",
        ],
    },
}

# コンテキスト付与テンプレート（文章の多様性を増やす）
CONTEXT_TEMPLATES = [
    "{sentence}",
    "ある日、{sentence}",
    "村人によると、{sentence}という",
    "旅人が驚いたことに、{sentence}",
    "この地方では当然のことだが、{sentence}",
]


def generate_base_tasks() -> List[Task]:
    """ベースタスクを生成する（各ルール20問 × ルール数）"""
    tasks = []
    task_id = 0

    for rule, sentence_sets in SENTENCES.items():
        for sentence in sentence_sets["consistent"]:
            tasks.append(Task(
                task_id=task_id,
                question=sentence,
                label=True,
                world_rule=rule,
            ))
            task_id += 1

        for sentence in sentence_sets["inconsistent"]:
            tasks.append(Task(
                task_id=task_id,
                question=sentence,
                label=False,
                world_rule=rule,
            ))
            task_id += 1

    return tasks


def generate_tasks(count: int = 100, seed: int = 42) -> List[Task]:
    """指定数のタスクを生成する（テンプレート増幅で拡張）"""
    rng = random.Random(seed)
    base_tasks = generate_base_tasks()

    if count <= len(base_tasks):
        rng.shuffle(base_tasks)
        return [Task(task_id=i, question=t.question, label=t.label, world_rule=t.world_rule)
                for i, t in enumerate(base_tasks[:count])]

    # テンプレート増幅
    augmented = []
    for base in base_tasks:
        for template in CONTEXT_TEMPLATES:
            augmented.append(Task(
                task_id=0,
                question=template.format(sentence=base.question),
                label=base.label,
                world_rule=base.world_rule,
            ))

    rng.shuffle(augmented)
    tasks = augmented[:count]

    # task_id を振り直す
    for i, t in enumerate(tasks):
        t.task_id = i

    return tasks


def format_prompt(task: Task) -> str:
    """タスクをプロンプト形式に変換する"""
    return (
        f"ワールドルール：{task.world_rule}\n"
        f"文章：「{task.question}」\n"
        f"この文章はワールドルールと矛盾しますか？\n"
        f"矛盾しない場合は「矛盾しない」、矛盾する場合は「矛盾する」とだけ答えてください。"
    )


def task_to_jsonl_record(task: Task) -> dict:
    """タスクをファインチューニング用JSONL形式に変換する"""
    return {
        "task_id": task.task_id,
        "prompt": format_prompt(task),
        "completion": "矛盾しない" if task.label else "矛盾する",
        "label": task.label,
        "world_rule": task.world_rule,
        "question": task.question,
    }


def main():
    parser = argparse.ArgumentParser(description="ワールドルール矛盾検出タスク生成")
    parser.add_argument("--count", type=int, default=100, help="生成するタスク数 (default: 100)")
    parser.add_argument("--output", type=str, default=None, help="出力先JSONLファイル (default: stdout表示)")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード (default: 42)")
    args = parser.parse_args()

    tasks = generate_tasks(count=args.count, seed=args.seed)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for task in tasks:
                f.write(json.dumps(task_to_jsonl_record(task), ensure_ascii=False) + "\n")
        consistent = sum(1 for t in tasks if t.label)
        inconsistent = sum(1 for t in tasks if not t.label)
        print(f"生成完了: {len(tasks)}問 → {args.output}")
        print(f"  矛盾しない: {consistent}問")
        print(f"  矛盾する:   {inconsistent}問")
    else:
        # stdout表示モード（従来互換）
        print(f"生成されたタスク数: {len(tasks)}")
        consistent = sum(1 for t in tasks if t.label)
        inconsistent = sum(1 for t in tasks if not t.label)
        print(f"  矛盾しない: {consistent}問")
        print(f"  矛盾する:   {inconsistent}問")
        print()
        print("サンプル（最初の3問）:")
        for t in tasks[:3]:
            print(f"  [{t.task_id}] {t.question[:30]}... | 正解: {'矛盾しない' if t.label else '矛盾する'}")


if __name__ == "__main__":
    main()
