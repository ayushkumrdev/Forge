"""SWE-micro — a graduated benchmark that discriminates between agent
configurations at small model scale.

SWE-bench is calibrated for frontier models; sub-10B models score near zero
on it, which produces no signal for comparing configurations. SWE-micro is
built to sit in the responsive band for 7B-class models while still being
real work on real repositories:

  T1  single edit             (add a function, fix a bug in one place)
  T2  several requirements     (multi-part change, still one file)
  T3  cross-file change        (touching imports / several modules)
  T4  repository-level task    (make a failing test pass, small refactor)

The ladder is deliberate: the first full sweep found T1 solved ~67% and
cross-file work solved 0/6 in every configuration — a floor that cannot
discriminate. T2 exists to fill that gap, holding the file count at one and
varying only how many requirements the request contains.

Each task materializes a fixture repository, gives the agent a plain-English
request, and scores it with a hidden pytest suite the agent never sees. The
checks are written against behaviour, not implementation, so any correct
solution passes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:
    id: str
    tier: int
    request: str
    files: dict[str, str]  # fixture repo: relative path -> content
    check: str  # hidden pytest module, written to test_hidden.py
    tags: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------
# T1 — single-file edits
# --------------------------------------------------------------------------

_T1_ADD_FUNCTION = Task(
    id="t1-add-function",
    tier=1,
    request=(
        "Add a function `slugify(text)` to strings.py that lowercases the text, "
        "replaces every run of non-alphanumeric characters with a single hyphen, "
        "and strips leading/trailing hyphens."
    ),
    files={
        "strings.py": (
            "import re\n\n\n"
            "def titlecase(text):\n"
            '    """Capitalize the first letter of every word."""\n'
            "    return re.sub(r'\\w+', lambda m: m.group(0).capitalize(), text)\n"
        ),
    },
    check=(
        "from strings import slugify\n\n\n"
        "def test_basic():\n"
        "    assert slugify('Hello World') == 'hello-world'\n\n\n"
        "def test_collapses_and_strips():\n"
        "    assert slugify('  **Hello,,, World!!  ') == 'hello-world'\n\n\n"
        "def test_existing_untouched():\n"
        "    from strings import titlecase\n"
        "    assert titlecase('ab cd') == 'Ab Cd'\n"
    ),
    tags=("create", "regex"),
)

_T1_FIX_BUG = Task(
    id="t1-fix-offbyone",
    tier=1,
    request=(
        "chunk(items, size) in batching.py drops the final partial batch. "
        "Fix it so every item is returned."
    ),
    files={
        "batching.py": (
            "def chunk(items, size):\n"
            '    """Split items into batches of `size`."""\n'
            "    out = []\n"
            "    for i in range(0, len(items) - size + 1, size):\n"
            "        out.append(items[i:i + size])\n"
            "    return out\n"
        ),
    },
    check=(
        "from batching import chunk\n\n\n"
        "def test_exact_multiple():\n"
        "    assert chunk([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]\n\n\n"
        "def test_partial_tail_kept():\n"
        "    assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\n\n\n"
        "def test_empty():\n"
        "    assert chunk([], 3) == []\n"
    ),
    tags=("bugfix",),
)

_T1_EDGE_CASE = Task(
    id="t1-guard-divzero",
    tier=1,
    request=(
        "average(values) in stats.py crashes on an empty list. Make it return "
        "0.0 for an empty list instead, without changing behaviour otherwise."
    ),
    files={
        "stats.py": (
            "def average(values):\n"
            '    """Arithmetic mean of a sequence of numbers."""\n'
            "    return sum(values) / len(values)\n"
        ),
    },
    check=(
        "from stats import average\n\n\n"
        "def test_empty_is_zero():\n"
        "    assert average([]) == 0.0\n\n\n"
        "def test_normal_unchanged():\n"
        "    assert average([1, 2, 3]) == 2\n"
    ),
    tags=("bugfix", "edge-case"),
)


# --------------------------------------------------------------------------
# T2 — several requirements, one file
#
# The gap the first full sweep exposed: T1 (one edit) was solved ~67% of the
# time and T3 (coordinate two files) was solved 0/6 in every configuration —
# a floor with no signal. These tasks keep the work inside a single file and
# vary only the number of requirements, isolating "did it cover the whole
# request" from "can it coordinate across modules".
# --------------------------------------------------------------------------

_T2_TWO_PART = Task(
    id="t2-add-and-use",
    tier=2,
    request=(
        "In prices.py: add a function `apply_discount(amount, percent)` that "
        "returns the amount reduced by that percentage, and then use it inside "
        "`checkout(items, percent)` so the returned total is discounted."
    ),
    files={
        "prices.py": (
            "def subtotal(items):\n"
            '    """Sum of price * qty for every item."""\n'
            "    return sum(i['price'] * i['qty'] for i in items)\n\n\n"
            "def checkout(items, percent):\n"
            '    """Final amount the customer pays."""\n'
            "    return subtotal(items)\n"
        ),
    },
    check=(
        "from prices import apply_discount, checkout\n\n"
        "ITEMS = [{'price': 50, 'qty': 2}]\n\n\n"
        "def test_discount_helper():\n"
        "    assert apply_discount(100, 10) == 90\n"
        "    assert apply_discount(100, 0) == 100\n\n\n"
        "def test_checkout_uses_it():\n"
        "    assert checkout(ITEMS, 10) == 90\n\n\n"
        "def test_checkout_without_discount():\n"
        "    assert checkout(ITEMS, 0) == 100\n"
    ),
    tags=("multi-requirement", "same-file"),
)

_T2_THREE_PART = Task(
    id="t2-three-guards",
    tier=2,
    request=(
        "Harden `parse_port(text)` in netconf.py with three changes: strip "
        "surrounding whitespace before parsing, raise ValueError for a port "
        "outside 1-65535, and return None instead of crashing when the text "
        "is not a number at all."
    ),
    files={
        "netconf.py": (
            "def parse_port(text):\n"
            '    """Parse a port number from configuration text."""\n'
            "    return int(text)\n"
        ),
    },
    check=(
        "import pytest\n\n"
        "from netconf import parse_port\n\n\n"
        "def test_strips_whitespace():\n"
        "    assert parse_port('  8080 ') == 8080\n\n\n"
        "def test_rejects_out_of_range():\n"
        "    with pytest.raises(ValueError):\n"
        "        parse_port('70000')\n"
        "    with pytest.raises(ValueError):\n"
        "        parse_port('0')\n\n\n"
        "def test_non_numeric_is_none():\n"
        "    assert parse_port('http') is None\n\n\n"
        "def test_normal_still_works():\n"
        "    assert parse_port('443') == 443\n"
    ),
    tags=("multi-requirement", "same-file", "edge-case"),
)

_T2_RENAME_LOCAL = Task(
    id="t2-rename-in-file",
    tier=2,
    request=(
        "In taskqueue.py, rename `push` to `enqueue` and `pop` to `dequeue`, and "
        "update every use of them inside that same file so nothing breaks."
    ),
    files={
        "taskqueue.py": (
            "class Queue:\n"
            "    def __init__(self):\n"
            "        self._items = []\n\n"
            "    def push(self, item):\n"
            "        self._items.append(item)\n\n"
            "    def pop(self):\n"
            "        return self._items.pop(0)\n\n"
            "    def drain(self):\n"
            "        out = []\n"
            "        while self._items:\n"
            "            out.append(self.pop())\n"
            "        return out\n\n\n"
            "def fill(queue, values):\n"
            "    for value in values:\n"
            "        queue.push(value)\n"
            "    return queue\n"
        ),
    },
    check=(
        "from queue import Queue, fill\n\n\n"
        "def test_renamed():\n"
        "    q = Queue()\n"
        "    assert hasattr(q, 'enqueue') and hasattr(q, 'dequeue')\n"
        "    assert not hasattr(q, 'push') and not hasattr(q, 'pop')\n\n\n"
        "def test_internal_uses_updated():\n"
        "    q = fill(Queue(), [1, 2, 3])\n"
        "    assert q.drain() == [1, 2, 3]\n"
    ),
    tags=("multi-requirement", "same-file", "refactor"),
)


# --------------------------------------------------------------------------
# T3 — cross-file changes
# --------------------------------------------------------------------------

_T3_ADD_AND_WIRE = Task(
    id="t3-wire-validator",
    tier=3,
    request=(
        "Add a `validate_email(address)` helper to validators.py that returns "
        "True only when the address contains exactly one '@' with non-empty "
        "text on both sides and a '.' after the '@'. Then use it in "
        "signup.py: register() must raise ValueError('invalid email') for a "
        "bad address before creating the user."
    ),
    files={
        "validators.py": (
            "def validate_username(name):\n"
            "    return bool(name) and name.isalnum()\n"
        ),
        "signup.py": (
            "from validators import validate_username\n\n"
            "USERS = []\n\n\n"
            "def register(username, email):\n"
            "    if not validate_username(username):\n"
            "        raise ValueError('invalid username')\n"
            "    USERS.append({'username': username, 'email': email})\n"
            "    return USERS[-1]\n"
        ),
    },
    check=(
        "import pytest\n\n"
        "import signup\nfrom validators import validate_email\n\n\n"
        "def setup_function():\n"
        "    signup.USERS.clear()\n\n\n"
        "def test_validator_accepts_good():\n"
        "    assert validate_email('a@b.com') is True\n\n\n"
        "def test_validator_rejects_bad():\n"
        "    assert validate_email('ab.com') is False\n"
        "    assert validate_email('@b.com') is False\n"
        "    assert validate_email('a@bcom') is False\n\n\n"
        "def test_register_rejects_bad_email():\n"
        "    with pytest.raises(ValueError):\n"
        "        signup.register('bob', 'not-an-email')\n"
        "    assert signup.USERS == []\n\n\n"
        "def test_register_still_works():\n"
        "    user = signup.register('bob', 'bob@example.com')\n"
        "    assert user['username'] == 'bob'\n"
    ),
    tags=("cross-file", "import"),
)

_T3_RENAME = Task(
    id="t3-rename-propagate",
    tier=3,
    request=(
        "Rename the function `calc` in engine.py to `compute_total`, and update "
        "every place that calls it so nothing is broken."
    ),
    files={
        "engine.py": (
            "def calc(items):\n"
            '    """Total price of the items."""\n'
            "    return sum(i['price'] * i['qty'] for i in items)\n"
        ),
        "cart.py": (
            "from engine import calc\n\n\n"
            "def cart_total(cart):\n"
            "    return calc(cart['items'])\n"
        ),
        "report.py": (
            "import engine\n\n\n"
            "def summary(orders):\n"
            "    return [engine.calc(o['items']) for o in orders]\n"
        ),
    },
    check=(
        "import engine\nfrom cart import cart_total\nfrom report import summary\n\n"
        "ITEMS = [{'price': 2, 'qty': 3}]\n\n\n"
        "def test_renamed():\n"
        "    assert hasattr(engine, 'compute_total')\n"
        "    assert not hasattr(engine, 'calc')\n\n\n"
        "def test_callers_updated():\n"
        "    assert cart_total({'items': ITEMS}) == 6\n"
        "    assert summary([{'items': ITEMS}]) == [6]\n"
    ),
    tags=("cross-file", "refactor"),
)


# --------------------------------------------------------------------------
# T4 — repository-level
# --------------------------------------------------------------------------

_T4_FAILING_TEST = Task(
    id="t4-make-suite-pass",
    tier=4,
    request=(
        "The test suite in this repository is failing. Run it, find the cause, "
        "and fix the source so every test passes. Do not modify the tests."
    ),
    files={
        "inventory.py": (
            "class Inventory:\n"
            "    def __init__(self):\n"
            "        self._items = {}\n\n"
            "    def add(self, name, qty=1):\n"
            "        self._items[name] = qty\n\n"
            "    def remove(self, name, qty=1):\n"
            "        self._items[name] -= qty\n\n"
            "    def count(self, name):\n"
            "        return self._items[name]\n"
        ),
        "test_inventory.py": (
            "from inventory import Inventory\n\n\n"
            "def test_add_accumulates():\n"
            "    inv = Inventory()\n"
            "    inv.add('apple', 2)\n"
            "    inv.add('apple', 3)\n"
            "    assert inv.count('apple') == 5\n\n\n"
            "def test_count_missing_is_zero():\n"
            "    assert Inventory().count('ghost') == 0\n\n\n"
            "def test_remove_floors_at_zero():\n"
            "    inv = Inventory()\n"
            "    inv.add('pen', 1)\n"
            "    inv.remove('pen', 5)\n"
            "    assert inv.count('pen') == 0\n"
        ),
    },
    check=(
        "from inventory import Inventory\n\n\n"
        "def test_add_accumulates():\n"
        "    inv = Inventory()\n"
        "    inv.add('apple', 2)\n"
        "    inv.add('apple', 3)\n"
        "    assert inv.count('apple') == 5\n\n\n"
        "def test_count_missing_is_zero():\n"
        "    assert Inventory().count('ghost') == 0\n\n\n"
        "def test_remove_floors_at_zero():\n"
        "    inv = Inventory()\n"
        "    inv.add('pen', 1)\n"
        "    inv.remove('pen', 5)\n"
        "    assert inv.count('pen') == 0\n"
    ),
    tags=("debug", "run-tests"),
)

_T4_FEATURE = Task(
    id="t4-add-cli-flag",
    tier=4,
    request=(
        "Add a --upper flag to the CLI in app.py: when passed, greet() output "
        "is uppercased. Keep the default behaviour identical."
    ),
    files={
        "app.py": (
            "import argparse\n\n\n"
            "def greet(name):\n"
            "    return f'hello, {name}'\n\n\n"
            "def build_parser():\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('name')\n"
            "    return parser\n\n\n"
            "def main(argv=None):\n"
            "    args = build_parser().parse_args(argv)\n"
            "    return greet(args.name)\n"
        ),
    },
    check=(
        "from app import main\n\n\n"
        "def test_default_unchanged():\n"
        "    assert main(['bob']) == 'hello, bob'\n\n\n"
        "def test_upper_flag():\n"
        "    assert main(['bob', '--upper']) == 'HELLO, BOB'\n"
    ),
    tags=("feature", "cli"),
)


SUITE: tuple[Task, ...] = (
    _T1_ADD_FUNCTION,
    _T1_FIX_BUG,
    _T1_EDGE_CASE,
    _T2_TWO_PART,
    _T2_THREE_PART,
    _T2_RENAME_LOCAL,
    _T3_ADD_AND_WIRE,
    _T3_RENAME,
    _T4_FAILING_TEST,
    _T4_FEATURE,
)


def tasks(tier: int | None = None, ids: list[str] | None = None) -> list[Task]:
    selected = list(SUITE)
    if tier is not None:
        selected = [t for t in selected if t.tier == tier]
    if ids:
        wanted = set(ids)
        selected = [t for t in selected if t.id in wanted]
    return selected
