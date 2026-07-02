#!/usr/bin/env python3
"""Author the task suites and write them as JSONL.

Tasks are authored here as Python literals (escaping-safe, reviewable) and
serialized to the JSONL files the harness loads. Rerun after editing:

    python tasks/generate_tasks.py

The committed JSONL is the artifact of record; this file is its source.
Every coding task includes a reference_solution and every agentic task a
reference_commands list — the test suite executes both to prove each task
is solvable and correctly specified.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.hermes import RUN_COMMANDS_TOOL

OUT_DIR = Path(__file__).resolve().parent


# ===========================================================================
# coding — executable tests, sandboxed
# ===========================================================================

CODING = [
    {
        "id": "rle_encode",
        "type": "code",
        "entry_point": "rle_encode",
        "prompt": (
            "Write a Python function rle_encode(s: str) -> str that run-length "
            "encodes a string: each maximal run of a character becomes the "
            "character followed by its count. Example: 'aaabbc' -> 'a3b2c1'. "
            "An empty string encodes to an empty string."
        ),
        "test_code": (
            "assert rle_encode('aaabbc') == 'a3b2c1'\n"
            "assert rle_encode('') == ''\n"
            "assert rle_encode('z') == 'z1'\n"
            "assert rle_encode('aabbaa') == 'a2b2a2'\n"
        ),
        "reference_solution": (
            "def rle_encode(s):\n"
            "    if not s:\n"
            "        return ''\n"
            "    out, prev, count = [], s[0], 1\n"
            "    for ch in s[1:]:\n"
            "        if ch == prev:\n"
            "            count += 1\n"
            "        else:\n"
            "            out.append(f'{prev}{count}')\n"
            "            prev, count = ch, 1\n"
            "    out.append(f'{prev}{count}')\n"
            "    return ''.join(out)\n"
        ),
    },
    {
        "id": "balanced_brackets",
        "type": "code",
        "entry_point": "is_balanced",
        "prompt": (
            "Write a Python function is_balanced(s: str) -> bool that returns "
            "True when every bracket in s — (), [], {} — is correctly matched "
            "and properly nested. Non-bracket characters are ignored."
        ),
        "test_code": (
            "assert is_balanced('([]{})') is True\n"
            "assert is_balanced('([)]') is False\n"
            "assert is_balanced('') is True\n"
            "assert is_balanced('((') is False\n"
            "assert is_balanced('a(b)c[d]') is True\n"
            "assert is_balanced(')(') is False\n"
        ),
        "reference_solution": (
            "def is_balanced(s):\n"
            "    pairs = {')': '(', ']': '[', '}': '{'}\n"
            "    stack = []\n"
            "    for ch in s:\n"
            "        if ch in '([{':\n"
            "            stack.append(ch)\n"
            "        elif ch in pairs:\n"
            "            if not stack or stack.pop() != pairs[ch]:\n"
            "                return False\n"
            "    return not stack\n"
        ),
    },
    {
        "id": "compare_semver",
        "type": "code",
        "entry_point": "compare_semver",
        "prompt": (
            "Write a Python function compare_semver(a: str, b: str) -> int for "
            "versions of the form 'MAJOR.MINOR.PATCH' (numeric components "
            "only). Return -1 if a < b, 0 if equal, 1 if a > b. Comparison is "
            "numeric per component, so '1.2.10' > '1.2.9'."
        ),
        "test_code": (
            "assert compare_semver('1.2.10', '1.2.9') == 1\n"
            "assert compare_semver('1.0.0', '1.0.0') == 0\n"
            "assert compare_semver('1.99.99', '2.0.0') == -1\n"
            "assert compare_semver('0.1.0', '0.0.9') == 1\n"
        ),
        "reference_solution": (
            "def compare_semver(a, b):\n"
            "    ta = tuple(int(x) for x in a.split('.'))\n"
            "    tb = tuple(int(x) for x in b.split('.'))\n"
            "    return (ta > tb) - (ta < tb)\n"
        ),
    },
    {
        "id": "flatten_dict",
        "type": "code",
        "entry_point": "flatten_dict",
        "prompt": (
            "Write a Python function flatten_dict(d: dict) -> dict that "
            "flattens arbitrarily nested dicts into a single level with "
            "dot-separated keys. Example: {'a': {'b': 1}, 'c': 2} -> "
            "{'a.b': 1, 'c': 2}. Non-dict values are kept as-is."
        ),
        "test_code": (
            "assert flatten_dict({'a': {'b': 1}, 'c': 2}) == {'a.b': 1, 'c': 2}\n"
            "assert flatten_dict({'a': {'b': {'c': 3}}, 'd': 4}) == {'a.b.c': 3, 'd': 4}\n"
            "assert flatten_dict({}) == {}\n"
            "assert flatten_dict({'x': [1, 2]}) == {'x': [1, 2]}\n"
        ),
        "reference_solution": (
            "def flatten_dict(d, prefix=''):\n"
            "    out = {}\n"
            "    for k, v in d.items():\n"
            "        key = f'{prefix}.{k}' if prefix else str(k)\n"
            "        if isinstance(v, dict) and v:\n"
            "            out.update(flatten_dict(v, key))\n"
            "        else:\n"
            "            out[key] = v\n"
            "    return out\n"
        ),
    },
    {
        "id": "topo_sort",
        "type": "code",
        "entry_point": "topo_sort",
        "prompt": (
            "Write a Python function topo_sort(graph: dict) -> list. `graph` "
            "maps every node to the list of nodes it depends on (all nodes "
            "appear as keys). Return an ordering in which every node appears "
            "after all of its dependencies. Raise ValueError if the graph "
            "contains a cycle."
        ),
        "test_code": (
            "g = {'a': ['b', 'c'], 'b': ['c'], 'c': []}\n"
            "order = topo_sort(g)\n"
            "assert sorted(order) == ['a', 'b', 'c']\n"
            "for node, deps in g.items():\n"
            "    for d in deps:\n"
            "        assert order.index(d) < order.index(node)\n"
            "try:\n"
            "    topo_sort({'x': ['y'], 'y': ['x']})\n"
            "    raise AssertionError('expected ValueError on cycle')\n"
            "except ValueError:\n"
            "    pass\n"
        ),
        "reference_solution": (
            "from collections import deque\n"
            "def topo_sort(graph):\n"
            "    indeg = {n: len(deps) for n, deps in graph.items()}\n"
            "    dependents = {n: [] for n in graph}\n"
            "    for n, deps in graph.items():\n"
            "        for d in deps:\n"
            "            dependents[d].append(n)\n"
            "    q = deque(sorted(n for n, k in indeg.items() if k == 0))\n"
            "    order = []\n"
            "    while q:\n"
            "        n = q.popleft()\n"
            "        order.append(n)\n"
            "        for m in dependents[n]:\n"
            "            indeg[m] -= 1\n"
            "            if indeg[m] == 0:\n"
            "                q.append(m)\n"
            "    if len(order) != len(graph):\n"
            "        raise ValueError('cycle detected')\n"
            "    return order\n"
        ),
    },
    {
        "id": "days_between",
        "type": "code",
        "entry_point": "days_between",
        "prompt": (
            "Write a Python function days_between(a: str, b: str) -> int that "
            "returns the absolute number of days between two ISO dates "
            "('YYYY-MM-DD'). Handle leap years correctly (use the standard "
            "library)."
        ),
        "test_code": (
            "assert days_between('2024-02-28', '2024-03-01') == 2\n"
            "assert days_between('2023-02-28', '2023-03-01') == 1\n"
            "assert days_between('2024-03-01', '2024-02-28') == 2\n"
            "assert days_between('2024-01-01', '2024-01-01') == 0\n"
        ),
        "reference_solution": (
            "from datetime import date\n"
            "def days_between(a, b):\n"
            "    return abs((date.fromisoformat(b) - date.fromisoformat(a)).days)\n"
        ),
    },
    {
        "id": "search_insert",
        "type": "code",
        "entry_point": "search_insert",
        "prompt": (
            "Write a Python function search_insert(nums: list, target: int) -> "
            "int. `nums` is sorted ascending with no duplicates. Return the "
            "index of target if present, else the index where it would be "
            "inserted to keep the list sorted. Must run in O(log n) — use "
            "binary search."
        ),
        "test_code": (
            "assert search_insert([1, 3, 5, 6], 5) == 2\n"
            "assert search_insert([1, 3, 5, 6], 2) == 1\n"
            "assert search_insert([1, 3, 5, 6], 7) == 4\n"
            "assert search_insert([], 3) == 0\n"
            "assert search_insert([1, 3, 5, 6], 0) == 0\n"
        ),
        "reference_solution": (
            "def search_insert(nums, target):\n"
            "    lo, hi = 0, len(nums)\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if nums[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid\n"
            "    return lo\n"
        ),
    },
    {
        "id": "group_anagrams",
        "type": "code",
        "entry_point": "group_anagrams",
        "prompt": (
            "Write a Python function group_anagrams(words: list) -> list that "
            "groups words that are anagrams of each other. Return a list of "
            "lists (group order and in-group order do not matter)."
        ),
        "test_code": (
            "res = group_anagrams(['eat', 'tea', 'tan', 'ate', 'nat', 'bat'])\n"
            "got = {frozenset(g) for g in res}\n"
            "want = {frozenset({'eat', 'tea', 'ate'}), frozenset({'tan', 'nat'}),\n"
            "        frozenset({'bat'})}\n"
            "assert got == want\n"
            "assert group_anagrams([]) == []\n"
        ),
        "reference_solution": (
            "def group_anagrams(words):\n"
            "    groups = {}\n"
            "    for w in words:\n"
            "        groups.setdefault(''.join(sorted(w)), []).append(w)\n"
            "    return list(groups.values())\n"
        ),
    },
    {
        "id": "lru_cache",
        "type": "code",
        "entry_point": "LRUCache",
        "prompt": (
            "Implement a Python class LRUCache with capacity given to "
            "__init__. Methods: get(key) -> value or -1 if absent (and marks "
            "the key most-recently used); put(key, value) inserts/updates and "
            "evicts the least-recently-used entry when over capacity. Both "
            "must be O(1) average."
        ),
        "test_code": (
            "c = LRUCache(2)\n"
            "c.put(1, 1)\n"
            "c.put(2, 2)\n"
            "assert c.get(1) == 1\n"
            "c.put(3, 3)\n"
            "assert c.get(2) == -1\n"
            "c.put(4, 4)\n"
            "assert c.get(1) == -1\n"
            "assert c.get(3) == 3\n"
            "assert c.get(4) == 4\n"
        ),
        "reference_solution": (
            "from collections import OrderedDict\n"
            "class LRUCache:\n"
            "    def __init__(self, capacity):\n"
            "        self.cap = capacity\n"
            "        self.d = OrderedDict()\n"
            "    def get(self, key):\n"
            "        if key not in self.d:\n"
            "            return -1\n"
            "        self.d.move_to_end(key)\n"
            "        return self.d[key]\n"
            "    def put(self, key, value):\n"
            "        if key in self.d:\n"
            "            self.d.move_to_end(key)\n"
            "        self.d[key] = value\n"
            "        if len(self.d) > self.cap:\n"
            "            self.d.popitem(last=False)\n"
        ),
    },
    {
        "id": "valid_ipv4",
        "type": "code",
        "entry_point": "valid_ipv4",
        "prompt": (
            "Write a Python function valid_ipv4(s: str) -> bool. A valid "
            "address has exactly four dot-separated decimal octets 0-255, "
            "digits only, and no leading zeros (so '01.1.1.1' is invalid but "
            "'0.1.1.1' is valid)."
        ),
        "test_code": (
            "assert valid_ipv4('192.168.1.1') is True\n"
            "assert valid_ipv4('256.1.1.1') is False\n"
            "assert valid_ipv4('01.1.1.1') is False\n"
            "assert valid_ipv4('1.1.1') is False\n"
            "assert valid_ipv4('0.0.0.0') is True\n"
            "assert valid_ipv4('1.2.3.4.5') is False\n"
            "assert valid_ipv4('a.b.c.d') is False\n"
        ),
        "reference_solution": (
            "def valid_ipv4(s):\n"
            "    parts = s.split('.')\n"
            "    if len(parts) != 4:\n"
            "        return False\n"
            "    for p in parts:\n"
            "        if not p.isdigit():\n"
            "            return False\n"
            "        if len(p) > 1 and p[0] == '0':\n"
            "            return False\n"
            "        if int(p) > 255:\n"
            "            return False\n"
            "    return True\n"
        ),
    },
]


# ===========================================================================
# math — numeric answers with tolerance
# ===========================================================================

MATH = [
    {"id": "chairs_4_weeks", "type": "numeric", "answer": 352,
     "prompt": "A workshop builds 14 chairs per weekday and 9 chairs per day "
               "on weekends. How many chairs does it build in exactly 4 weeks?"},
    {"id": "linear_equation", "type": "numeric", "answer": -12,
     "prompt": "Solve for x: 3x + 7 = 2x - 5."},
    {"id": "definite_integral", "type": "numeric", "answer": 16,
     "prompt": "Compute the definite integral of 6x^2 dx from x = 0 to x = 2."},
    {"id": "dice_sum_9", "type": "numeric", "answer": 0.11111, "rel_tol": 0.02,
     "prompt": "Two fair six-sided dice are rolled. What is the probability "
               "that their sum equals 9? Give the answer as a decimal."},
    {"id": "compound_interest", "type": "numeric", "answer": 5624.32, "rel_tol": 0.001,
     "prompt": "You invest $5000 at 4% annual interest, compounded yearly. "
               "What is the balance after 3 years, in dollars?"},
    {"id": "derivative_at_2", "type": "numeric", "answer": 8,
     "prompt": "Let f(x) = x^3 - 4x. What is f'(2)?"},
    {"id": "gcd_252_198", "type": "numeric", "answer": 18,
     "prompt": "What is the greatest common divisor of 252 and 198?"},
    {"id": "sum_first_40_odd", "type": "numeric", "answer": 1600,
     "prompt": "What is the sum of the first 40 positive odd numbers?"},
    {"id": "det_2x2", "type": "numeric", "answer": 5,
     "prompt": "Compute the determinant of the 2x2 matrix [[2, 3], [1, 4]]."},
    {"id": "log2_one_32nd", "type": "numeric", "answer": -5,
     "prompt": "Evaluate log base 2 of (1/32)."},
]


# ===========================================================================
# stem_engineering — ME / EE / physics, numeric with tolerance
# ===========================================================================

STEM = [
    {"id": "spring_frequency", "type": "numeric", "answer": 3.183, "rel_tol": 0.02,
     "prompt": "A 2 kg mass hangs from a spring with stiffness k = 800 N/m. "
               "What is the natural frequency of vertical oscillation, in Hz?"},
    {"id": "series_resistor_power", "type": "numeric", "answer": 6,
     "prompt": "A 12 V source drives three resistors in series: 4 ohm, 6 ohm, "
               "and 2 ohm. How much power is dissipated in the 6 ohm resistor, "
               "in watts?"},
    {"id": "cantilever_deflection", "type": "numeric", "answer": 1.667, "rel_tol": 0.02,
     "prompt": "A cantilever beam of length 2 m carries a 1000 N point load at "
               "its free end. E = 200 GPa and I = 8e-6 m^4. What is the tip "
               "deflection, in millimeters?"},
    {"id": "voltage_divider", "type": "numeric", "answer": 3,
     "prompt": "A voltage divider uses R1 = 10 kohm (top) and R2 = 5 kohm "
               "(bottom) across a 9 V supply. What is the output voltage "
               "across R2, in volts?"},
    {"id": "water_heating_energy", "type": "numeric", "answer": 502320, "rel_tol": 0.01,
     "prompt": "How much energy, in joules, is needed to heat 2 kg of water "
               "from 20 C to 80 C? Use c = 4186 J/(kg*K)."},
    {"id": "projectile_range", "type": "numeric", "answer": 91.74, "rel_tol": 0.02,
     "prompt": "A projectile is launched at 30 m/s at 45 degrees above "
               "horizontal on level ground. Ignoring air resistance and using "
               "g = 9.81 m/s^2, what is its range, in meters?"},
    {"id": "rc_time_constant", "type": "numeric", "answer": 1.0,
     "prompt": "What is the time constant, in seconds, of an RC circuit with "
               "R = 10 kohm and C = 100 microfarads?"},
    {"id": "gear_ratio_rpm", "type": "numeric", "answer": 600,
     "prompt": "A 20-tooth gear spinning at 1800 rpm drives a 60-tooth gear. "
               "What is the driven gear's speed, in rpm?"},
    {"id": "hydrostatic_pressure", "type": "numeric", "answer": 98100, "rel_tol": 0.01,
     "prompt": "What is the gauge pressure, in pascals, at a depth of 10 m in "
               "fresh water? Use rho = 1000 kg/m^3 and g = 9.81 m/s^2."},
    {"id": "rod_axial_stress", "type": "numeric", "answer": 101.86, "rel_tol": 0.02,
     "prompt": "A steel rod of diameter 25 mm carries a 50 kN axial tensile "
               "load. What is the normal stress in the rod, in MPa?"},
]


# ===========================================================================
# reasoning — multiple choice
# ===========================================================================

REASONING = [
    {"id": "syllogism", "type": "mcq", "answer": "C",
     "prompt": "All bloops are razzies. Some razzies are lazzies. Which "
               "statement MUST be true?",
     "options": ["Some bloops are lazzies", "All razzies are bloops",
                 "All bloops are razzies", "Some lazzies are bloops"]},
    {"id": "number_sequence", "type": "mcq", "answer": "B",
     "prompt": "What is the next number in the sequence 2, 6, 12, 20, 30, ...?",
     "options": ["40", "42", "44", "36"]},
    {"id": "knights_knaves", "type": "mcq", "answer": "D",
     "prompt": "On an island, knights always tell the truth and knaves always "
               "lie. Person A says: 'We are both knaves' (about A and B). "
               "What are A and B?",
     "options": ["Both knights", "Both knaves",
                 "A is a knight, B is a knave", "A is a knave, B is a knight"]},
    {"id": "affirming_consequent", "type": "mcq", "answer": "C",
     "prompt": "If it rains, the ground gets wet. The ground is wet. What can "
               "you VALIDLY conclude about whether it rained?",
     "options": ["It rained", "It did not rain",
                 "It cannot be determined from the given information",
                 "The ground is always wet"]},
    {"id": "clock_angle", "type": "mcq", "answer": "B",
     "prompt": "What is the angle between the hour and minute hands of an "
               "analog clock at exactly 3:30?",
     "options": ["60 degrees", "75 degrees", "90 degrees", "105 degrees"]},
    {"id": "mislabeled_boxes", "type": "mcq", "answer": "C",
     "prompt": "Three boxes are labeled 'Apples', 'Oranges', and 'Apples and "
               "Oranges'. Every label is wrong. You may draw one fruit from "
               "one box to deduce the contents of all three. Which box should "
               "you draw from?",
     "options": ["The one labeled 'Apples'", "The one labeled 'Oranges'",
                 "The one labeled 'Apples and Oranges'",
                 "Any box works equally well"]},
    {"id": "race_order", "type": "mcq", "answer": "B",
     "prompt": "In a race, Dana finished ahead of Anna, Anna finished ahead "
               "of Ben, and Ben finished ahead of Carl. Who finished third?",
     "options": ["Anna", "Ben", "Carl", "Dana"]},
    {"id": "binary_addition", "type": "mcq", "answer": "A",
     "prompt": "What is 1011 + 110 in binary?",
     "options": ["10001", "1111", "10010", "1101"]},
    {"id": "sock_pigeonhole", "type": "mcq", "answer": "C",
     "prompt": "A drawer contains many socks in exactly 3 colors. How many "
               "socks must you take out, without looking, to GUARANTEE a "
               "matching pair?",
     "options": ["2", "3", "4", "5"]},
    {"id": "odd_one_out", "type": "mcq", "answer": "C",
     "prompt": "Which number does not belong with the others: 3, 5, 11, 14, 17?",
     "options": ["3", "11", "14", "17"]},
]


# ===========================================================================
# tool_use — Hermes tool-call precision
# ===========================================================================

WEATHER = {"type": "function", "function": {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"},
        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    }, "required": ["city", "units"]}}}

TICKET = {"type": "function", "function": {
    "name": "create_ticket",
    "description": "Create an issue ticket in the tracker.",
    "parameters": {"type": "object", "properties": {
        "title": {"type": "string"},
        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        "labels": {"type": "array", "items": {"type": "string"}},
    }, "required": ["title", "priority"]}}}

SEARCH = {"type": "function", "function": {
    "name": "search_repo",
    "description": "Search the repository for code matching a query.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string"},
        "file_glob": {"type": "string",
                      "description": "Glob restricting which files to search."},
    }, "required": ["query", "file_glob"]}}}

MEETING = {"type": "function", "function": {
    "name": "schedule_meeting",
    "description": "Schedule a meeting on the calendar.",
    "parameters": {"type": "object", "properties": {
        "title": {"type": "string"},
        "start_iso": {"type": "string",
                      "description": "Start time, ISO 8601, e.g. 2026-07-10T14:00:00"},
        "duration_minutes": {"type": "integer"},
    }, "required": ["title", "start_iso", "duration_minutes"]}}}

CONVERT = {"type": "function", "function": {
    "name": "convert_units",
    "description": "Convert a value between units.",
    "parameters": {"type": "object", "properties": {
        "value": {"type": "number"},
        "from_unit": {"type": "string",
                      "enum": ["miles", "kilometers", "pounds", "kilograms",
                               "celsius", "fahrenheit"]},
        "to_unit": {"type": "string",
                    "enum": ["miles", "kilometers", "pounds", "kilograms",
                             "celsius", "fahrenheit"]},
    }, "required": ["value", "from_unit", "to_unit"]}}}

TOOL_USE = [
    {"id": "weather_single", "type": "tool_call", "tools": [WEATHER],
     "prompt": "What's the current weather in Tokyo? Give it to me in celsius.",
     "expect": {"calls": [{"name": "get_weather",
                           "arguments_contain": {"city": "tokyo"},
                           "arguments_equal": {"units": "celsius"}}]}},
    {"id": "weather_parallel", "type": "tool_call", "tools": [WEATHER],
     "prompt": "I'm deciding between Berlin and Madrid for the weekend. Check "
               "the current weather in both cities, in celsius.",
     "expect": {"calls": [
         {"name": "get_weather", "arguments_contain": {"city": "berlin"},
          "arguments_equal": {"units": "celsius"}},
         {"name": "get_weather", "arguments_contain": {"city": "madrid"},
          "arguments_equal": {"units": "celsius"}}]}},
    {"id": "ticket_urgent", "type": "tool_call", "tools": [TICKET, SEARCH],
     "prompt": "Production checkout is failing for all users since the last "
               "deploy. Open a ticket titled exactly 'Checkout page 500 "
               "errors' and set the priority appropriately — this is urgent.",
     "expect": {"calls": [{"name": "create_ticket",
                           "arguments_equal": {"title": "Checkout page 500 errors",
                                                "priority": "high"}}]}},
    {"id": "no_tool_needed", "type": "tool_call", "tools": [WEATHER, TICKET],
     "prompt": "What does HTTP status code 404 mean?",
     "expect": {"no_call": True}},
    {"id": "clarify_before_call", "type": "tool_call", "tools": [MEETING],
     "prompt": "Schedule a meeting with the infra team.",
     "expect": {"no_call": True, "must_ask_question": True}},
    {"id": "convert_exact_args", "type": "tool_call", "tools": [CONVERT],
     "prompt": "Use the unit converter to convert 26.2 miles to kilometers.",
     "expect": {"calls": [{"name": "convert_units",
                           "arguments_equal": {"value": 26.2,
                                                "from_unit": "miles",
                                                "to_unit": "kilometers"}}]}},
    {"id": "search_definition", "type": "tool_call",
     "tools": [SEARCH, RUN_COMMANDS_TOOL],
     "prompt": "Find where the function handle_login is defined in this "
               "repo's Python files.",
     "expect": {"calls": [{"name": "search_repo",
                           "arguments_contain": {"query": "handle_login",
                                                  "file_glob": ".py"}}]}},
    {"id": "disk_usage_command", "type": "tool_call", "tools": [RUN_COMMANDS_TOOL],
     "prompt": "Check how much disk space is free on this machine.",
     "expect": {"calls": [{"name": "run_commands",
                           "arguments_contain": {"commands": "df"}}]}},
    {"id": "meeting_exact_args", "type": "tool_call", "tools": [MEETING],
     "prompt": "Schedule 'Sprint retro' for 2026-07-10T14:00:00, lasting 45 "
               "minutes.",
     "expect": {"calls": [{"name": "schedule_meeting",
                           "arguments_equal": {"title": "Sprint retro",
                                                "start_iso": "2026-07-10T14:00:00",
                                                "duration_minutes": 45}}]}},
    {"id": "no_tool_from_memory", "type": "tool_call", "tools": [CONVERT, WEATHER],
     "prompt": "Roughly how many feet are in a mile? Just answer from memory.",
     "expect": {"no_call": True}},
]


# ===========================================================================
# instruction_following — mechanical constraints
# ===========================================================================

INSTRUCTION = [
    {"id": "three_bullets", "type": "instruction",
     "prompt": "List exactly 3 advantages of git rebase over git merge, as "
               "bullet points. Each bullet must start with '- '. Output only "
               "the three bullets — no introduction, no closing text.",
     "checks": [{"kind": "bullet_count", "value": 3}]},
    {"id": "json_config_only", "type": "instruction",
     "prompt": "Output a JSON object — and nothing else, no code fences, no "
               "commentary — with exactly these keys: name (string), port "
               "(integer), debug (boolean). Choose sensible values for a "
               "local dev web server.",
     "checks": [{"kind": "json_only", "keys": ["name", "port", "debug"]},
                {"kind": "starts_with", "value": "{"}]},
    {"id": "tcp_forty_words", "type": "instruction",
     "prompt": "Explain the TCP three-way handshake in at most 40 words. You "
               "must mention SYN.",
     "checks": [{"kind": "max_words", "value": 40},
                {"kind": "must_include", "values": ["SYN"]}]},
    {"id": "compiler_haiku", "type": "instruction",
     "prompt": "Write a haiku about compilers. Output exactly three lines — "
               "no title, no commentary, nothing else.",
     "checks": [{"kind": "line_count", "value": 3}]},
    {"id": "lowercase_kernel", "type": "instruction",
     "prompt": "describe what the linux kernel does, in one sentence, using "
               "only lowercase letters. no capital letters may appear "
               "anywhere in your reply.",
     "checks": [{"kind": "lowercase_only"}]},
    {"id": "marker_and_done", "type": "instruction",
     "prompt": "Explain what a segmentation fault is in exactly two "
               "sentences. Include the exact string ERROR_CODE_42 somewhere "
               "in your answer, and end your reply with exactly: Done.",
     "checks": [{"kind": "must_include_exact", "values": ["ERROR_CODE_42"]},
                {"kind": "ends_with", "value": "Done."}]},
    {"id": "five_numbered_steps", "type": "instruction",
     "prompt": "Give a numbered list of exactly 5 steps to create and "
               "activate a Python virtual environment on Linux. Each step on "
               "its own line, starting with '1.' through '5.'. No other "
               "numbered lines.",
     "checks": [{"kind": "regex_count", "pattern": "^\\d+\\.", "value": 5}]},
    {"id": "forbidden_word", "type": "instruction",
     "prompt": "Explain what a SQL index is and why it speeds up queries — "
               "but you are not allowed to use the word 'database' anywhere "
               "in your answer.",
     "checks": [{"kind": "must_include", "values": ["index"]},
                {"kind": "must_not_include", "values": ["database"]}]},
    {"id": "csv_header_exact", "type": "instruction",
     "prompt": "Output a CSV table with the header row exactly 'id,name,email' "
               "followed by two data rows for fictional users. The very first "
               "line of your reply must be the header — nothing before it, "
               "no code fences.",
     "checks": [{"kind": "starts_with", "value": "id,name,email"}]},
    {"id": "two_paragraphs", "type": "instruction",
     "prompt": "Describe the difference between processes and threads in "
               "exactly two paragraphs, separated by a single blank line. No "
               "headings, no lists.",
     "checks": [{"kind": "paragraph_count", "value": 2}]},
]


# ===========================================================================
# planning_agentic — multi-turn episodes in the VirtualShell
# ===========================================================================

AGENTIC = [
    {"id": "scaffold_project", "type": "agentic", "max_turns": 6,
     "prompt": "Create a project skeleton in the current directory: "
               "directories 'src' and 'tests', an empty file src/__init__.py, "
               "and a README.md whose content is '# Demo'.",
     "initial_files": {},
     "goals": [
         {"kind": "dir_exists", "path": "src"},
         {"kind": "dir_exists", "path": "tests"},
         {"kind": "file_exists", "path": "src/__init__.py"},
         {"kind": "file_contains", "path": "README.md", "text": "# Demo"}],
     "reference_commands": [
         ["mkdir -p src tests", "touch src/__init__.py",
          "echo '# Demo' > README.md"]]},
    {"id": "find_todo_delete", "type": "agentic", "max_turns": 6,
     "prompt": "Exactly one file under notes/ contains the text 'TODO'. Find "
               "it and delete it. Leave every other file untouched.",
     "initial_files": {"notes/a.txt": "buy milk\n",
                       "notes/b.txt": "TODO: remove this file\n",
                       "notes/c.txt": "done\n"},
     "goals": [
         {"kind": "file_absent", "path": "notes/b.txt"},
         {"kind": "file_exists", "path": "notes/a.txt"},
         {"kind": "file_exists", "path": "notes/c.txt"}],
     "reference_commands": [["grep -r TODO notes"], ["rm notes/b.txt"]]},
    {"id": "fix_config_flag", "type": "agentic", "max_turns": 6,
     "prompt": "In config.env, change DEBUG to false. Keep the PORT setting "
               "exactly as it is. The file must end up containing both "
               "settings.",
     "initial_files": {"config.env": "DEBUG=true\nPORT=8080\n"},
     "goals": [
         {"kind": "file_contains", "path": "config.env", "text": "DEBUG=false"},
         {"kind": "file_contains", "path": "config.env", "text": "PORT=8080"},
         {"kind": "file_not_contains", "path": "config.env", "text": "DEBUG=true"}],
     "reference_commands": [
         ["cat config.env"],
         ["echo 'DEBUG=false' > config.env && echo 'PORT=8080' >> config.env"],
         ["cat config.env"]]},
    {"id": "incident_log", "type": "agentic", "max_turns": 6,
     "prompt": "One of the log files in logs/ contains a FATAL error. Find "
               "that file and copy it to incident.log in the current "
               "directory.",
     "initial_files": {"logs/app1.log": "started ok\nready\n",
                       "logs/app2.log": "started\nFATAL: db connection lost\n",
                       "logs/app3.log": "started ok\n"},
     "goals": [
         {"kind": "file_exists", "path": "incident.log"},
         {"kind": "file_contains", "path": "incident.log", "text": "FATAL"}],
     "reference_commands": [["grep -r FATAL logs"],
                             ["cp logs/app2.log incident.log"]]},
    {"id": "cleanup_tmp_files", "type": "agentic", "max_turns": 6,
     "prompt": "Delete every file with the .tmp extension anywhere in the "
               "workspace. Do not delete anything else.",
     "initial_files": {"build/a.tmp": "x\n", "build/b.tmp": "y\n",
                       "build/keep.txt": "keep me\n", "cache.tmp": "z\n"},
     "goals": [
         {"kind": "file_absent", "path": "build/a.tmp"},
         {"kind": "file_absent", "path": "build/b.tmp"},
         {"kind": "file_absent", "path": "cache.tmp"},
         {"kind": "file_exists", "path": "build/keep.txt"}],
     "reference_commands": [["ls", "ls build"],
                             ["rm build/a.tmp build/b.tmp cache.tmp"]]},
    {"id": "release_notes_version", "type": "agentic", "max_turns": 6,
     "prompt": "Read the version number stored in version.txt, then create "
               "release/notes.txt whose content is the line 'release ' "
               "followed by that version (for example, 'release 1.0.0' if "
               "the version were 1.0.0).",
     "initial_files": {"version.txt": "2.7.1\n"},
     "goals": [{"kind": "file_contains", "path": "release/notes.txt",
                "text": "release 2.7.1"}],
     "reference_commands": [
         ["cat version.txt"],
         ["mkdir -p release && echo 'release 2.7.1' > release/notes.txt"]]},
    {"id": "reorganize_module", "type": "agentic", "max_turns": 6,
     "prompt": "Move helper.py into src/utils/ (create directories as "
               "needed). The file's content must be preserved and the "
               "original location must be gone.",
     "initial_files": {"helper.py": "def helper():\n    return 42\n"},
     "goals": [
         {"kind": "file_contains", "path": "src/utils/helper.py",
          "text": "def helper"},
         {"kind": "file_absent", "path": "helper.py"}],
     "reference_commands": [["mkdir -p src/utils && mv helper.py src/utils"]]},
    {"id": "backup_then_append", "type": "agentic", "max_turns": 6,
     "prompt": "First create a backup copy of data.csv named data.csv.bak. "
               "Then append the row '2,bob' to data.csv. The backup must "
               "keep the original content only.",
     "initial_files": {"data.csv": "id,name\n1,alice\n"},
     "goals": [
         {"kind": "file_contains", "path": "data.csv.bak", "text": "1,alice"},
         {"kind": "file_not_contains", "path": "data.csv.bak", "text": "2,bob"},
         {"kind": "file_contains", "path": "data.csv", "text": "2,bob"},
         {"kind": "file_contains", "path": "data.csv", "text": "1,alice"}],
     "reference_commands": [["cp data.csv data.csv.bak"],
                             ["echo '2,bob' >> data.csv"]]},
]


SUITE_DATA = {
    "coding": CODING,
    "math": MATH,
    "stem_engineering": STEM,
    "reasoning": REASONING,
    "tool_use": TOOL_USE,
    "instruction_following": INSTRUCTION,
    "planning_agentic": AGENTIC,
}


def main() -> None:
    total = 0
    for suite, tasks in SUITE_DATA.items():
        ids = [t["id"] for t in tasks]
        assert len(ids) == len(set(ids)), f"duplicate ids in {suite}"
        path = OUT_DIR / f"{suite}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for task in tasks:
                fh.write(json.dumps(task, ensure_ascii=False) + "\n")
        print(f"wrote {path.name}: {len(tasks)} tasks")
        total += len(tasks)
    print(f"total: {total} tasks")


if __name__ == "__main__":
    main()
