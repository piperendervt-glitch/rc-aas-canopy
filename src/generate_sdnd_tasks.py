"""
generate_sdnd_tasks.py
sdnd-devのT2（バグ修正）・T5（エラーハンドリング）タスクをTRUSSフォーマットに変換

T2：「このコードにバグがあるか/ないか」の二値判定
T5：「このコードは安全か/危険か」の二値判定

出力：data/sdnd_task_pool.jsonl
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# =============================================================
# T2：バグ修正タスク（consistent=バグなし, inconsistent=バグあり）
# =============================================================

T2_RULE = "正しいPythonコードの規則：変数は使用前に定義され、インデックスは範囲内であり、型は正しく使われ、ロジックに誤りがない"

T2_PAIRS = [
    # (バグなしコード, バグありコード)
    (
        "def average(nums):\n    if not nums:\n        return 0\n    return sum(nums) / len(nums)",
        "def average(nums):\n    return sum(nums) / len(nums)",
    ),
    (
        "def find_max(lst):\n    if not lst:\n        return None\n    result = lst[0]\n    for x in lst[1:]:\n        if x > result:\n            result = x\n    return result",
        "def find_max(lst):\n    result = lst[0]\n    for x in lst:\n        if x > result:\n            result = x\n    return result",
    ),
    (
        "def factorial(n):\n    if n < 0:\n        raise ValueError('negative')\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n)",
    ),
    (
        "def reverse_string(s):\n    return s[::-1]",
        "def reverse_string(s):\n    return s[::1]",
    ),
    (
        "def is_palindrome(s):\n    s = s.lower().strip()\n    return s == s[::-1]",
        "def is_palindrome(s):\n    return s == s[::-1]",
    ),
    (
        "def count_words(text):\n    if not text or not text.strip():\n        return 0\n    return len(text.split())",
        "def count_words(text):\n    return len(text.split())",
    ),
    (
        "def safe_divide(a, b):\n    if b == 0:\n        return None\n    return a / b",
        "def safe_divide(a, b):\n    return a / b",
    ),
    (
        "def get_element(lst, idx):\n    if 0 <= idx < len(lst):\n        return lst[idx]\n    return None",
        "def get_element(lst, idx):\n    return lst[idx]",
    ),
    (
        "def fibonacci(n):\n    if n <= 0:\n        return 0\n    if n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b",
        "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    ),
    (
        "def flatten(nested):\n    result = []\n    for item in nested:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result",
        "def flatten(nested):\n    result = []\n    for item in nested:\n        if type(item) == list:\n            result.append(flatten(item))\n        else:\n            result.append(item)\n    return result",
    ),
    (
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid\n    return -1",
    ),
    (
        "def merge_dicts(d1, d2):\n    result = d1.copy()\n    result.update(d2)\n    return result",
        "def merge_dicts(d1, d2):\n    result = d1\n    result.update(d2)\n    return result",
    ),
    (
        "def remove_duplicates(lst):\n    seen = set()\n    result = []\n    for x in lst:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result",
        "def remove_duplicates(lst):\n    result = []\n    for x in lst:\n        if x not in result:\n            result.append(x)\n    return result",
    ),
    (
        "def clamp(value, lo, hi):\n    return max(lo, min(hi, value))",
        "def clamp(value, lo, hi):\n    return min(lo, max(hi, value))",
    ),
    (
        "def is_even(n):\n    return n % 2 == 0",
        "def is_even(n):\n    return n % 2 == 1",
    ),
    (
        "def sum_range(start, end):\n    return sum(range(start, end + 1))",
        "def sum_range(start, end):\n    return sum(range(start, end))",
    ),
    (
        "def pop_safe(lst):\n    if lst:\n        return lst.pop()\n    return None",
        "def pop_safe(lst):\n    return lst.pop()",
    ),
    (
        "def str_to_int(s):\n    try:\n        return int(s)\n    except (ValueError, TypeError):\n        return None",
        "def str_to_int(s):\n    return int(s)",
    ),
    (
        "def unique_count(lst):\n    return len(set(lst))",
        "def unique_count(lst):\n    return len(lst)",
    ),
    (
        "def swap(lst, i, j):\n    if 0 <= i < len(lst) and 0 <= j < len(lst):\n        lst[i], lst[j] = lst[j], lst[i]",
        "def swap(lst, i, j):\n    lst[i], lst[j] = lst[j], lst[i]",
    ),
    (
        "def first_or_default(lst, default=None):\n    return lst[0] if lst else default",
        "def first_or_default(lst, default=None):\n    return lst[0]",
    ),
    (
        "def power(base, exp):\n    if exp < 0:\n        return 1 / power(base, -exp)\n    if exp == 0:\n        return 1\n    return base * power(base, exp - 1)",
        "def power(base, exp):\n    if exp == 0:\n        return 1\n    return base * power(base, exp - 1)",
    ),
    (
        "def chunk(lst, size):\n    if size <= 0:\n        raise ValueError('size must be positive')\n    return [lst[i:i+size] for i in range(0, len(lst), size)]",
        "def chunk(lst, size):\n    return [lst[i:i+size] for i in range(0, len(lst), size)]",
    ),
    (
        "def safe_key(d, key, default=None):\n    return d.get(key, default)",
        "def safe_key(d, key, default=None):\n    return d[key]",
    ),
    (
        "def zip_longest(a, b, fill=None):\n    n = max(len(a), len(b))\n    result = []\n    for i in range(n):\n        x = a[i] if i < len(a) else fill\n        y = b[i] if i < len(b) else fill\n        result.append((x, y))\n    return result",
        "def zip_longest(a, b, fill=None):\n    n = max(len(a), len(b))\n    result = []\n    for i in range(n):\n        result.append((a[i], b[i]))\n    return result",
    ),
]

# =============================================================
# T5：エラーハンドリングタスク（consistent=安全, inconsistent=危険）
# =============================================================

T5_RULE = "安全なPythonコードの規則：外部入力は検証され、例外は適切に処理され、リソースは確実に解放され、セキュリティ上の脆弱性がない"

T5_PAIRS = [
    # (安全コード, 危険コード)
    (
        "def read_file(path):\n    try:\n        with open(path, 'r') as f:\n            return f.read()\n    except (FileNotFoundError, PermissionError):\n        return None",
        "def read_file(path):\n    f = open(path, 'r')\n    data = f.read()\n    return data",
    ),
    (
        "def parse_json(text):\n    try:\n        return json.loads(text)\n    except json.JSONDecodeError:\n        return None",
        "def parse_json(text):\n    return json.loads(text)",
    ),
    (
        "import subprocess\ndef run_cmd(cmd):\n    result = subprocess.run(\n        cmd, shell=False, capture_output=True,\n        timeout=30, text=True\n    )\n    return result.stdout",
        "import os\ndef run_cmd(cmd):\n    return os.system(cmd)",
    ),
    (
        "def query_db(conn, user_id):\n    cursor = conn.cursor()\n    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))\n    return cursor.fetchone()",
        "def query_db(conn, user_id):\n    cursor = conn.cursor()\n    cursor.execute(f'SELECT * FROM users WHERE id = {user_id}')\n    return cursor.fetchone()",
    ),
    (
        "import html\ndef render(user_input):\n    safe = html.escape(user_input)\n    return f'<div>{safe}</div>'",
        "def render(user_input):\n    return f'<div>{user_input}</div>'",
    ),
    (
        "def download(url):\n    try:\n        resp = requests.get(url, timeout=10)\n        resp.raise_for_status()\n        return resp.text\n    except requests.RequestException:\n        return None",
        "def download(url):\n    return requests.get(url).text",
    ),
    (
        "def write_file(path, data):\n    try:\n        with open(path, 'w') as f:\n            f.write(data)\n        return True\n    except IOError:\n        return False",
        "def write_file(path, data):\n    open(path, 'w').write(data)",
    ),
    (
        "def parse_int(value):\n    try:\n        return int(value)\n    except (ValueError, TypeError):\n        return 0",
        "def parse_int(value):\n    return int(value)",
    ),
    (
        "import re\ndef validate_email(email):\n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return bool(re.match(pattern, email))",
        "def validate_email(email):\n    return '@' in email",
    ),
    (
        "def connect_db(config):\n    conn = None\n    try:\n        conn = db.connect(**config)\n        return conn\n    except db.Error as e:\n        if conn:\n            conn.close()\n        raise",
        "def connect_db(config):\n    conn = db.connect(**config)\n    return conn",
    ),
    (
        "import hashlib\ndef hash_password(password, salt):\n    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)",
        "import hashlib\ndef hash_password(password):\n    return hashlib.md5(password.encode()).hexdigest()",
    ),
    (
        "def load_config(path):\n    try:\n        with open(path) as f:\n            config = yaml.safe_load(f)\n        return config or {}\n    except Exception:\n        return {}",
        "def load_config(path):\n    with open(path) as f:\n        return yaml.load(f, Loader=yaml.FullLoader)",
    ),
    (
        "import tempfile\ndef temp_write(data):\n    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:\n        f.write(data)\n        return f.name",
        "def temp_write(data):\n    path = '/tmp/output.txt'\n    open(path, 'w').write(data)\n    return path",
    ),
    (
        "def safe_eval(expr):\n    import ast\n    try:\n        tree = ast.parse(expr, mode='eval')\n        return eval(compile(tree, '<string>', 'eval'))\n    except (SyntaxError, ValueError):\n        return None",
        "def safe_eval(expr):\n    return eval(expr)",
    ),
    (
        "def get_env(key, default=''):\n    import os\n    return os.environ.get(key, default)",
        "def get_env(key):\n    import os\n    return os.environ[key]",
    ),
    (
        "import secrets\ndef gen_token(n=32):\n    return secrets.token_hex(n)",
        "import random\ndef gen_token(n=32):\n    return ''.join(random.choice('abcdef0123456789') for _ in range(n))",
    ),
    (
        "def divide(a, b):\n    if not isinstance(b, (int, float)) or b == 0:\n        raise ValueError('invalid divisor')\n    return a / b",
        "def divide(a, b):\n    return a / b",
    ),
    (
        "def read_csv(path):\n    rows = []\n    try:\n        with open(path, newline='', encoding='utf-8') as f:\n            reader = csv.reader(f)\n            for row in reader:\n                rows.append(row)\n    except (FileNotFoundError, csv.Error):\n        pass\n    return rows",
        "def read_csv(path):\n    with open(path) as f:\n        return list(csv.reader(f))",
    ),
    (
        "def sanitize_path(base, user_path):\n    import os\n    full = os.path.realpath(os.path.join(base, user_path))\n    if not full.startswith(os.path.realpath(base)):\n        raise ValueError('path traversal')\n    return full",
        "def sanitize_path(base, user_path):\n    import os\n    return os.path.join(base, user_path)",
    ),
    (
        "def retry(func, max_retries=3):\n    for i in range(max_retries):\n        try:\n            return func()\n        except Exception:\n            if i == max_retries - 1:\n                raise\n            continue",
        "def retry(func):\n    while True:\n        try:\n            return func()\n        except Exception:\n            pass",
    ),
    (
        "import logging\ndef process(data):\n    try:\n        return transform(data)\n    except Exception as e:\n        logging.error('process failed: %s', e)\n        return None",
        "def process(data):\n    try:\n        return transform(data)\n    except Exception:\n        pass",
    ),
    (
        "def lock_resource(lock, func):\n    lock.acquire()\n    try:\n        return func()\n    finally:\n        lock.release()",
        "def lock_resource(lock, func):\n    lock.acquire()\n    result = func()\n    lock.release()\n    return result",
    ),
    (
        "def validate_age(value):\n    try:\n        age = int(value)\n        if not 0 <= age <= 150:\n            return None\n        return age\n    except (ValueError, TypeError):\n        return None",
        "def validate_age(value):\n    return int(value)",
    ),
    (
        "def send_request(url, data):\n    try:\n        resp = requests.post(url, json=data, timeout=10)\n        resp.raise_for_status()\n        return resp.json()\n    except (requests.RequestException, ValueError):\n        return None",
        "def send_request(url, data):\n    resp = requests.post(url, json=data)\n    return resp.json()",
    ),
    (
        "def thread_pool_exec(tasks, max_workers=4):\n    from concurrent.futures import ThreadPoolExecutor\n    results = []\n    with ThreadPoolExecutor(max_workers=max_workers) as pool:\n        futures = [pool.submit(t) for t in tasks]\n        for f in futures:\n            try:\n                results.append(f.result(timeout=30))\n            except Exception:\n                results.append(None)\n    return results",
        "def thread_pool_exec(tasks):\n    import threading\n    results = []\n    threads = [threading.Thread(target=lambda: results.append(t())) for t in tasks]\n    for t in threads:\n        t.start()\n    return results",
    ),
]


def generate_tasks():
    """T2・T5タスクを生成"""
    tasks = []
    task_id = 0

    # T2: バグ修正タスク
    for correct, buggy in T2_PAIRS:
        # consistent (バグなし)
        tasks.append({
            "task_id": task_id,
            "task_type": "T2",
            "rule": T2_RULE,
            "text": correct,
            "label": "consistent",
            "label_bool": True,
        })
        task_id += 1
        # inconsistent (バグあり)
        tasks.append({
            "task_id": task_id,
            "task_type": "T2",
            "rule": T2_RULE,
            "text": buggy,
            "label": "inconsistent",
            "label_bool": False,
        })
        task_id += 1

    # T5: エラーハンドリングタスク
    for safe, unsafe in T5_PAIRS:
        # consistent (安全)
        tasks.append({
            "task_id": task_id,
            "task_type": "T5",
            "rule": T5_RULE,
            "text": safe,
            "label": "consistent",
            "label_bool": True,
        })
        task_id += 1
        # inconsistent (危険)
        tasks.append({
            "task_id": task_id,
            "task_type": "T5",
            "rule": T5_RULE,
            "text": unsafe,
            "label": "inconsistent",
            "label_bool": False,
        })
        task_id += 1

    return tasks


def main():
    tasks = generate_tasks()

    # シャッフル（再現性あり）
    rng = random.Random(42)
    rng.shuffle(tasks)

    # task_id振り直し
    for i, t in enumerate(tasks):
        t["task_id"] = i

    output_path = DATA_DIR / "sdnd_task_pool.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # 統計
    t2_count = sum(1 for t in tasks if t["task_type"] == "T2")
    t5_count = sum(1 for t in tasks if t["task_type"] == "T5")
    consistent = sum(1 for t in tasks if t["label"] == "consistent")
    inconsistent = sum(1 for t in tasks if t["label"] == "inconsistent")

    print(f"=== sdnd T2/T5 task pool ===")
    print(f"Total: {len(tasks)}")
    print(f"  T2 (bug detection):      {t2_count} ({t2_count//2} consistent + {t2_count//2} inconsistent)")
    print(f"  T5 (error handling):     {t5_count} ({t5_count//2} consistent + {t5_count//2} inconsistent)")
    print(f"  consistent total:        {consistent}")
    print(f"  inconsistent total:      {inconsistent}")
    print(f"\nSaved to: {output_path}")

    # サンプル5問表示
    print(f"\n--- Sample 5 ---")
    for t in tasks[:5]:
        code_preview = t["text"][:60].replace("\n", " | ")
        print(f"\n  [{t['task_id']}] {t['task_type']} / {t['label']}")
        print(f"  rule: {t['rule'][:50]}...")
        print(f"  code: {code_preview}...")


if __name__ == "__main__":
    main()
