"""
generate_sdnd_all_tasks.py
sdnd-dev全51タスクをTRUSS二値判定フォーマットに変換

各タスクの"before"コード（問題あり）= inconsistent
手動で作成した"after"コード（改善済み）= consistent

出力：data/sdnd_all_tasks.jsonl
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# カテゴリ別ルール定義
RULES = {
    "documentation": "適切にドキュメントされたPythonコードの規則：関数にはdocstringがあり、引数と戻り値の説明があり、型ヒントが付与されている",
    "bugfix": "正しいPythonコードの規則：変数は使用前に定義され、インデックスは範囲内であり、型は正しく使われ、ロジックに誤りがない",
    "optimization": "効率的なPythonコードの規則：内包表記やジェネレータを活用し、冗長なループを避け、標準ライブラリを適切に使う",
    "naming": "適切な命名のPythonコードの規則：変数名は意味を持ち、snake_caseに準拠し、PEP8のスタイルガイドに従う",
    "error_handling": "安全なPythonコードの規則：例外は適切に処理され、エラー時にデフォルト値を返し、入力は検証される",
    "security": "セキュリティ要件を満たすPythonコードの規則：eval/execを使わず、入力をサニタイズし、安全な代替実装を用いる",
    "theater": "構造化されたPythonコードの規則：データは辞書で管理され、パースにはエラー処理があり、フォーマットは柔軟に対応する",
    "practical": "実用的なPythonコードの規則：標準ライブラリを活用し、入力バリデーションがあり、堅牢な実装である",
    "creative": "ユーザーフレンドリーなPythonコードの規則：エラーメッセージは具体的で、出力は読みやすく、コンテキストに応じた動作をする",
    "humaneval": "正しく動作するPythonコードの規則：仕様通りの入出力を返し、エッジケースに対応し、テストを全てパスする",
    "humaneval_pure": "正しく動作するPythonコードの規則：仕様通りの入出力を返し、エッジケースに対応し、テストを全てパスする",
}

# 全51タスク定義: (task_id, name, category, before_code, after_code)
TASKS = [
    # T1: ドキュメント追加
    ("T1", "documentation",
     "def add(a, b):\n    return a + b",
     'def add(a, b):\n    """2つの数値を加算して返す。\n\n    Args:\n        a: 加算する数値\n        b: 加算する数値\n\n    Returns:\n        a + b の結果\n    """\n    return a + b'),

    # T2: バグ修正（インデックスエラー）
    ("T2", "bugfix",
     "def get_last(lst):\n    return lst[len(lst)]",
     "def get_last(lst):\n    return lst[len(lst) - 1]"),

    # T3: ループ最適化（リスト内包表記）
    ("T3", "optimization",
     "def squares(n):\n    result = []\n    for i in range(n):\n        result.append(i ** 2)\n    return result",
     "def squares(n):\n    return [i ** 2 for i in range(n)]"),

    # T4: 変数名改善
    ("T4", "naming",
     "def calc(x, y, z):\n    a = x * y\n    b = a + z\n    return b",
     "def calc_total(price, tax_rate, discount):\n    subtotal = price * tax_rate\n    total = subtotal + discount\n    return total"),

    # T5: エラーハンドリング追加
    ("T5", "error_handling",
     "def divide(a, b):\n    return a / b",
     "def divide(a, b):\n    if b == 0:\n        raise ZeroDivisionError('divisor cannot be zero')\n    return a / b"),

    # T6: PEP8準拠
    ("T6", "naming",
     "def calculateTotalPrice(itemPrice,taxRate):\n  total=itemPrice+itemPrice*taxRate\n  return total",
     "def calculate_total_price(item_price, tax_rate):\n    total = item_price + item_price * tax_rate\n    return total"),

    # T7: 型ヒント追加
    ("T7", "documentation",
     "def greet(name, times):\n    return name * times\n\ndef average(numbers):\n    return sum(numbers) / len(numbers)",
     "def greet(name: str, times: int) -> str:\n    \"\"\"名前を指定回数繰り返す。\"\"\"\n    return name * times\n\ndef average(numbers: list[float]) -> float:\n    \"\"\"数値リストの平均を返す。\"\"\"\n    return sum(numbers) / len(numbers)"),

    # T8: 関数長制限（分割）
    ("T8", "optimization",
     "def process_data(data):\n    validated = []\n    for item in data:\n        if isinstance(item, dict):\n            if 'name' in item:\n                if len(item['name']) > 0:\n                    validated.append(item)\n    transformed = []\n    for item in validated:\n        new_item = {}\n        for key, value in item.items():\n            new_item[key.lower()] = str(value)\n        transformed.append(new_item)\n    result = []\n    for item in transformed:\n        result.append(item)\n    return result",
     "def validate_items(data):\n    return [item for item in data\n            if isinstance(item, dict) and item.get('name')]\n\ndef transform_items(items):\n    return [{k.lower(): str(v) for k, v in item.items()}\n            for item in items]\n\ndef process_data(data):\n    validated = validate_items(data)\n    return transform_items(validated)"),

    # T9: セキュリティ脆弱性修正
    ("T9", "security",
     "def calculate(expression):\n    return eval(expression)\n\ndef run_code(code):\n    exec(code)\n\ndef load_data(s):\n    return eval(s)",
     "import ast\nimport json\n\ndef calculate(expression):\n    tree = ast.parse(expression, mode='eval')\n    return eval(compile(tree, '<string>', 'eval'))\n\ndef run_code(code):\n    raise NotImplementedError('exec is not allowed')\n\ndef load_data(s):\n    return json.loads(s)"),

    # T10: ユニットテスト追加
    ("T10", "documentation",
     "def multiply(a, b):\n    return a * b",
     "def multiply(a, b):\n    return a * b\n\nimport unittest\n\nclass TestMultiply(unittest.TestCase):\n    def test_positive(self):\n        assert multiply(2, 3) == 6\n    def test_zero(self):\n        assert multiply(0, 5) == 0\n    def test_negative(self):\n        assert multiply(-1, 3) == -3"),

    # T11: ログ出力JSON統一
    ("T11", "optimization",
     "def process(data):\n    print('processing started')\n    result = data * 2\n    print('processing done:', result)\n    return result",
     "import logging\nimport json\n\nlogger = logging.getLogger(__name__)\n\ndef process(data):\n    logger.info(json.dumps({'event': 'processing_started'}))\n    result = data * 2\n    logger.info(json.dumps({'event': 'processing_done', 'result': result}))\n    return result"),

    # T12: ログレベル動的変更
    ("T12", "error_handling",
     "import logging\n\ndef setup_logger():\n    logger = logging.getLogger('app')\n    logger.setLevel(logging.DEBUG)\n    return logger",
     "import logging\n\ndef setup_logger(level='DEBUG'):\n    logger = logging.getLogger('app')\n    logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))\n    return logger"),

    # T13: import文の整理
    ("T13", "optimization",
     "import os\nimport sys\nimport json\nimport re\nimport math\n\ndef greet(name):\n    return f'Hello, {name}!'",
     "def greet(name):\n    return f'Hello, {name}!'"),

    # T14: 設定文字列パース
    ("T14", "error_handling",
     "def parse_config(config_str):\n    result = {}\n    for line in config_str.split('\\n'):\n        key, value = line.split('=')\n        result[key] = value\n    return result",
     "def parse_config(config_str):\n    \"\"\"設定文字列をパースして辞書を返す。\"\"\"\n    result = {}\n    for line in config_str.strip().split('\\n'):\n        if not line.strip() or '=' not in line:\n            continue\n        try:\n            key, value = line.split('=', 1)\n            result[key.strip()] = value.strip()\n        except ValueError:\n            continue\n    return result"),

    # T15: クラスを関数に分割
    ("T15", "optimization",
     "class Calculator:\n    def add(self, a, b):\n        return a + b\n    def multiply(self, a, b):\n        return a * b",
     "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b"),

    # T16: 三項演算子
    ("T16", "optimization",
     "def check_age(age):\n    if age >= 18:\n        status = 'adult'\n    else:\n        status = 'minor'\n    return status",
     "def check_age(age):\n    return 'adult' if age >= 18 else 'minor'"),

    # T17: argparse
    ("T17", "optimization",
     "def parse_args(args):\n    host = 'localhost'\n    port = 8080\n    for arg in args:\n        if arg.startswith('--host='):\n            host = arg.split('=')[1]\n        elif arg.startswith('--port='):\n            port = int(arg.split('=')[1])\n    return host, port",
     "import argparse\n\ndef parse_args(args=None):\n    parser = argparse.ArgumentParser()\n    parser.add_argument('--host', default='localhost')\n    parser.add_argument('--port', type=int, default=8080)\n    return parser.parse_args(args)"),

    # T18: try-exceptでデフォルト値
    ("T18", "error_handling",
     "def safe_int(value):\n    return int(value)",
     "def safe_int(value, default=0):\n    try:\n        return int(value)\n    except (ValueError, TypeError):\n        return default"),

    # T19: ジェネレータ関数
    ("T19", "optimization",
     "def get_even_numbers(numbers):\n    result = []\n    for n in numbers:\n        if n % 2 == 0:\n            result.append(n)\n    return result",
     "def get_even_numbers(numbers):\n    for n in numbers:\n        if n % 2 == 0:\n            yield n"),

    # T20: デコレータ共通化
    ("T20", "optimization",
     "def add(a, b):\n    print('calling add')\n    result = a + b\n    print('result:', result)\n    return result\n\ndef multiply(a, b):\n    print('calling multiply')\n    result = a * b\n    print('result:', result)\n    return result",
     "def log_call(func):\n    def wrapper(*args, **kwargs):\n        print(f'calling {func.__name__}')\n        result = func(*args, **kwargs)\n        print(f'result: {result}')\n        return result\n    return wrapper\n\n@log_call\ndef add(a, b):\n    return a + b\n\n@log_call\ndef multiply(a, b):\n    return a * b"),

    # T21: 辞書内包表記
    ("T21", "optimization",
     "def make_squares_dict(numbers):\n    result = {}\n    for n in numbers:\n        result[n] = n ** 2\n    return result",
     "def make_squares_dict(numbers):\n    return {n: n ** 2 for n in numbers}"),

    # T22: ガード節
    ("T22", "optimization",
     "def process_order(order):\n    if order is not None:\n        if order.get('status') == 'active':\n            if order.get('items'):\n                if len(order['items']) > 0:\n                    return sum(item['price'] for item in order['items'])\n    return 0",
     "def process_order(order):\n    if order is None:\n        return 0\n    if order.get('status') != 'active':\n        return 0\n    if not order.get('items'):\n        return 0\n    return sum(item['price'] for item in order['items'])"),

    # T23: プロパティデコレータ
    ("T23", "documentation",
     "import math\n\nclass Circle:\n    def __init__(self, radius):\n        self.radius = radius\n    def get_area(self):\n        return math.pi * self.radius ** 2\n    def get_circumference(self):\n        return 2 * math.pi * self.radius",
     "import math\n\nclass Circle:\n    def __init__(self, radius):\n        self.radius = radius\n\n    @property\n    def area(self):\n        return math.pi * self.radius ** 2\n\n    @property\n    def circumference(self):\n        return 2 * math.pi * self.radius"),

    # T24: ログフォーマットカスタマイズ
    ("T24", "error_handling",
     "def format_log(message):\n    return '[LOG] ' + message",
     "def format_log(message, level='INFO', fmt='[{level}] {message}'):\n    return fmt.format(level=level, message=message)"),

    # T25: enumerate使用
    ("T25", "optimization",
     "def index_items(items):\n    result = []\n    i = 0\n    for item in items:\n        result.append((i, item))\n        i += 1\n    return result",
     "def index_items(items):\n    return list(enumerate(items))"),

    # T26: シナリオテンプレート生成
    ("T26", "theater",
     "def create_scenario():\n    return 'Scene 1: Hello'",
     "def create_scenario(title='Untitled', characters=None, scenes=None):\n    scenario = {'title': title, 'characters': characters or [], 'scenes': scenes or []}\n    return scenario"),

    # T27: キャラクター設定パーサー
    ("T27", "theater",
     "def parse_character(text):\n    parts = text.split(',')\n    return {'name': parts[0], 'role': parts[1]}",
     "def parse_character(text):\n    parts = [p.strip() for p in text.split(',')]\n    if len(parts) < 2:\n        raise ValueError('name and role are required')\n    return {'name': parts[0], 'role': parts[1]}"),

    # T28: セリフフォーマッター
    ("T28", "theater",
     "def format_dialogue(name, line):\n    return name + ': ' + line",
     "def format_dialogue(name, line, fmt='{name}: {line}'):\n    return fmt.format(name=name, line=line)"),

    # T29: CSV→辞書変換
    ("T29", "practical",
     "def parse_csv(text):\n    lines = text.split('\\n')\n    headers = lines[0].split(',')\n    result = []\n    for line in lines[1:]:\n        values = line.split(',')\n        result.append(dict(zip(headers, values)))\n    return result",
     "import csv\nimport io\n\ndef parse_csv(text):\n    reader = csv.DictReader(io.StringIO(text))\n    return list(reader)"),

    # T30: 日付フォーマット統一
    ("T30", "practical",
     "def format_date(year, month, day):\n    return str(year) + '/' + str(month) + '/' + str(day)",
     "from datetime import datetime\n\ndef format_date(year, month, day):\n    return datetime(year, month, day).strftime('%Y/%m/%d')"),

    # T31: リトライデコレータ
    ("T31", "practical",
     "def fetch_data(url):\n    for i in range(3):\n        try:\n            return do_request(url)\n        except Exception:\n            if i == 2:\n                raise",
     "import functools\n\ndef retry(max_retries=3):\n    def decorator(func):\n        @functools.wraps(func)\n        def wrapper(*args, **kwargs):\n            for i in range(max_retries):\n                try:\n                    return func(*args, **kwargs)\n                except Exception:\n                    if i == max_retries - 1:\n                        raise\n        return wrapper\n    return decorator\n\n@retry(max_retries=3)\ndef fetch_data(url):\n    return do_request(url)"),

    # T32: バリデーション関数
    ("T32", "practical",
     "def register_user(name, age, email):\n    return {'name': name, 'age': age, 'email': email}",
     "def register_user(name, age, email):\n    if not isinstance(name, str) or not name.strip():\n        raise ValueError('name must be a non-empty string')\n    if not isinstance(age, int) or age < 0 or age > 150:\n        raise ValueError('age must be an integer between 0 and 150')\n    if not isinstance(email, str) or '@' not in email:\n        raise ValueError('email must contain @')\n    return {'name': name.strip(), 'age': age, 'email': email.strip()}"),

    # T33: エラーメッセージ改善
    ("T33", "creative",
     "def validate_input(value):\n    if not value:\n        raise ValueError('Error: invalid')\n    if len(value) > 100:\n        raise ValueError('Error: too long')\n    return value",
     "def validate_input(value):\n    if not value:\n        raise ValueError('Input is required. Please provide a non-empty value.')\n    if len(value) > 100:\n        raise ValueError(f'Input is too long ({len(value)} chars). Maximum is 100 characters.')\n    return value"),

    # T34: 挨拶文生成
    ("T34", "creative",
     "def greet():\n    return 'Hello'",
     "from datetime import datetime\n\ndef greet(name=''):\n    hour = datetime.now().hour\n    if hour < 12:\n        greeting = 'Good morning'\n    elif hour < 18:\n        greeting = 'Good afternoon'\n    else:\n        greeting = 'Good evening'\n    if name:\n        return f'{greeting}, {name}!'\n    return f'{greeting}!'"),

    # T35: ヘルプテキスト改善
    ("T35", "creative",
     "def show_help():\n    print('Usage: tool [options]')\n    print('Options: -h, -v')",
     "def show_help():\n    usage = [\n        'Usage: tool [options]',\n        '',\n        'Options:',\n        '  -h, --help     Show this help message',\n        '  -v, --version  Show version information',\n        '',\n        'Examples:',\n        '  tool --help',\n        '  tool -v',\n    ]\n    return '\\n'.join(usage)"),

    # T36: HE Two Sum
    ("T36", "humaneval",
     "def two_sum(nums, target):\n    pass",
     "def two_sum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        comp = target - n\n        if comp in seen:\n            return [seen[comp], i]\n        seen[n] = i\n    return []"),

    # T37: HE FizzBuzz
    ("T37", "humaneval",
     "def fizzbuzz(n):\n    pass",
     "def fizzbuzz(n):\n    result = []\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            result.append('FizzBuzz')\n        elif i % 3 == 0:\n            result.append('Fizz')\n        elif i % 5 == 0:\n            result.append('Buzz')\n        else:\n            result.append(str(i))\n    return result"),

    # T38: HE 回文判定
    ("T38", "humaneval",
     "def is_palindrome(s):\n    pass",
     "def is_palindrome(s):\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]"),

    # T39: HE フィボナッチ
    ("T39", "humaneval",
     "def fibonacci(n):\n    pass",
     "def fibonacci(n):\n    if n <= 0:\n        return 0\n    if n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"),

    # T40: HE リスト平坦化
    ("T40", "humaneval",
     "def flatten(lst):\n    pass",
     "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result"),

    # T41: HE 母音カウント
    ("T41", "humaneval",
     "def count_vowels(s):\n    pass",
     "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')"),

    # T42: HE 重複除去（順序保持）
    ("T42", "humaneval",
     "def remove_duplicates(lst):\n    pass",
     "def remove_duplicates(lst):\n    seen = set()\n    result = []\n    for x in lst:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result"),

    # T43: HE 最大公約数
    ("T43", "humaneval",
     "def gcd(a, b):\n    pass",
     "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a"),

    # T44: HE-Pure 括弧チェック
    ("T44", "humaneval_pure",
     "def is_valid_brackets(s):\n    pass",
     "def is_valid_brackets(s):\n    stack = []\n    pairs = {')': '(', ']': '[', '}': '{'}\n    for c in s:\n        if c in '([{':\n            stack.append(c)\n        elif c in pairs:\n            if not stack or stack[-1] != pairs[c]:\n                return False\n            stack.pop()\n    return len(stack) == 0"),

    # T45: HE-Pure ローマ数字→整数
    ("T45", "humaneval_pure",
     "def roman_to_int(s):\n    pass",
     "def roman_to_int(s):\n    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n    result = 0\n    for i in range(len(s)):\n        if i + 1 < len(s) and values[s[i]] < values[s[i + 1]]:\n            result -= values[s[i]]\n        else:\n            result += values[s[i]]\n    return result"),

    # T46: HE-Pure 最大部分配列和
    ("T46", "humaneval_pure",
     "def max_subarray_sum(nums):\n    pass",
     "def max_subarray_sum(nums):\n    if not nums:\n        return 0\n    max_sum = current = nums[0]\n    for n in nums[1:]:\n        current = max(n, current + n)\n        max_sum = max(max_sum, current)\n    return max_sum"),

    # T47: HE-Pure ソート済みマージ
    ("T47", "humaneval_pure",
     "def merge_sorted(list1, list2):\n    pass",
     "def merge_sorted(list1, list2):\n    result = []\n    i = j = 0\n    while i < len(list1) and j < len(list2):\n        if list1[i] <= list2[j]:\n            result.append(list1[i])\n            i += 1\n        else:\n            result.append(list2[j])\n            j += 1\n    result.extend(list1[i:])\n    result.extend(list2[j:])\n    return result"),

    # T48: HE-Pure 行列転置
    ("T48", "humaneval_pure",
     "def transpose(matrix):\n    pass",
     "def transpose(matrix):\n    if not matrix:\n        return []\n    return [list(row) for row in zip(*matrix)]"),

    # T49: HE-Pure 文字列圧縮
    ("T49", "humaneval_pure",
     "def compress(s):\n    pass",
     "def compress(s):\n    if not s:\n        return ''\n    result = []\n    count = 1\n    for i in range(1, len(s)):\n        if s[i] == s[i - 1]:\n            count += 1\n        else:\n            result.append(s[i - 1] + str(count))\n            count = 1\n    result.append(s[-1] + str(count))\n    return ''.join(result)"),

    # T50: HE-Pure 素数判定
    ("T50", "humaneval_pure",
     "def is_prime(n):\n    pass",
     "def is_prime(n):\n    if n < 2:\n        return False\n    if n < 4:\n        return True\n    if n % 2 == 0 or n % 3 == 0:\n        return False\n    i = 5\n    while i * i <= n:\n        if n % i == 0 or n % (i + 2) == 0:\n            return False\n        i += 6\n    return True"),

    # T51: HE-Pure 単語出現頻度
    ("T51", "humaneval_pure",
     "def word_count(text):\n    pass",
     "def word_count(text):\n    counts = {}\n    for word in text.lower().split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts"),
]

# sdnd-dev task_pool.jsonのメタデータ
TASK_META = {
    "T1": ("documentation", "easy", "docstring addition"),
    "T2": ("bugfix", "easy", "index out of bounds"),
    "T3": ("optimization", "easy", "list comprehension"),
    "T4": ("naming", "medium", "variable naming"),
    "T5": ("error_handling", "easy", "zero division"),
    "T6": ("naming", "medium", "PEP8 compliance"),
    "T7": ("documentation", "medium", "type hints"),
    "T8": ("optimization", "hard", "function splitting"),
    "T9": ("security", "hard", "eval/exec removal"),
    "T10": ("documentation", "medium", "unit tests"),
    "T11": ("optimization", "medium", "JSON logging"),
    "T12": ("error_handling", "medium", "dynamic log level"),
    "T13": ("optimization", "medium", "import cleanup"),
    "T14": ("error_handling", "medium", "config parsing"),
    "T15": ("optimization", "easy", "class to functions"),
    "T16": ("optimization", "easy", "ternary operator"),
    "T17": ("optimization", "medium", "argparse"),
    "T18": ("error_handling", "easy", "try-except default"),
    "T19": ("optimization", "medium", "generator"),
    "T20": ("optimization", "hard", "decorator"),
    "T21": ("optimization", "easy", "dict comprehension"),
    "T22": ("optimization", "medium", "guard clause"),
    "T23": ("documentation", "medium", "property decorator"),
    "T24": ("error_handling", "medium", "log format"),
    "T25": ("optimization", "easy", "enumerate"),
    "T26": ("theater", "medium", "scenario template"),
    "T27": ("theater", "medium", "character parser"),
    "T28": ("theater", "easy", "dialogue formatter"),
    "T29": ("practical", "medium", "CSV to dict"),
    "T30": ("practical", "easy", "date format"),
    "T31": ("practical", "hard", "retry decorator"),
    "T32": ("practical", "medium", "validation"),
    "T33": ("creative", "medium", "error messages"),
    "T34": ("creative", "easy", "greeting"),
    "T35": ("creative", "medium", "help text"),
    "T36": ("humaneval", "easy", "two sum"),
    "T37": ("humaneval", "easy", "fizzbuzz"),
    "T38": ("humaneval", "easy", "palindrome"),
    "T39": ("humaneval", "easy", "fibonacci"),
    "T40": ("humaneval", "medium", "flatten"),
    "T41": ("humaneval", "easy", "vowel count"),
    "T42": ("humaneval", "easy", "remove duplicates"),
    "T43": ("humaneval", "easy", "GCD"),
    "T44": ("humaneval_pure", "medium", "bracket matching"),
    "T45": ("humaneval_pure", "medium", "roman to int"),
    "T46": ("humaneval_pure", "medium", "max subarray"),
    "T47": ("humaneval_pure", "easy", "merge sorted"),
    "T48": ("humaneval_pure", "easy", "transpose"),
    "T49": ("humaneval_pure", "medium", "string compression"),
    "T50": ("humaneval_pure", "easy", "prime check"),
    "T51": ("humaneval_pure", "easy", "word count"),
}


def generate():
    tasks = []
    task_id = 0

    for sdnd_id, category, before_code, after_code in TASKS:
        meta = TASK_META[sdnd_id]
        rule = RULES[category]

        # consistent (改善済みコード)
        tasks.append({
            "task_id": task_id,
            "sdnd_task_id": sdnd_id,
            "task_type": category,
            "difficulty": meta[1],
            "sdnd_name": meta[2],
            "rule": rule,
            "text": after_code,
            "label": "consistent",
            "label_bool": True,
        })
        task_id += 1

        # inconsistent (問題ありコード)
        tasks.append({
            "task_id": task_id,
            "sdnd_task_id": sdnd_id,
            "task_type": category,
            "difficulty": meta[1],
            "sdnd_name": meta[2],
            "rule": rule,
            "text": before_code,
            "label": "inconsistent",
            "label_bool": False,
        })
        task_id += 1

    return tasks


def main():
    tasks = generate()

    # シャッフル
    rng = random.Random(42)
    rng.shuffle(tasks)
    for i, t in enumerate(tasks):
        t["task_id"] = i

    output_path = DATA_DIR / "sdnd_all_tasks.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # 統計
    categories = {}
    for t in tasks:
        cat = t["task_type"]
        categories[cat] = categories.get(cat, 0) + 1

    print(f"=== sdnd-dev all 51 tasks -> TRUSS format ===")
    print(f"Total: {len(tasks)} ({len(tasks)//2} pairs)")
    print(f"\nBy category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat:<20} {count} ({count//2} pairs)")

    consistent = sum(1 for t in tasks if t["label"] == "consistent")
    inconsistent = sum(1 for t in tasks if t["label"] == "inconsistent")
    print(f"\n  consistent:   {consistent}")
    print(f"  inconsistent: {inconsistent}")

    # サンプル5問
    print(f"\n--- Sample 5 ---")
    for t in tasks[:5]:
        code_preview = t["text"][:60].replace("\n", " | ")
        print(f"\n  [{t['task_id']}] {t['sdnd_task_id']} ({t['task_type']}/{t['difficulty']}) {t['label']}")
        print(f"  code: {code_preview}...")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
