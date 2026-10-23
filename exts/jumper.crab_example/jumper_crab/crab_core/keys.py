"""Isaac Sim's keyboard, in the controller's key names.

The controller reads keys by the names in `controller/vocabulary.json`
(`key_w`, `key_left_ctrl`, `keypad_2`, ...). That file gives each key's browser
`KeyboardEvent.code`, and carb's `KeyboardInput` names follow from it by a fixed
rule, so the table is derived rather than typed: a key the vocabulary adds later
works here without an edit, and one this rule cannot translate is reported.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

_SPECIAL = {
    "Space": "SPACE",
    "Semicolon": "SEMICOLON",
    "Escape": "ESCAPE",
    "ArrowUp": "UP",
    "ArrowDown": "DOWN",
    "ArrowLeft": "LEFT",
    "ArrowRight": "RIGHT",
    "ControlLeft": "LEFT_CONTROL",
    "ControlRight": "RIGHT_CONTROL",
    "ShiftLeft": "LEFT_SHIFT",
    "ShiftRight": "RIGHT_SHIFT",
    "AltLeft": "LEFT_ALT",
    "AltRight": "RIGHT_ALT",
    "NumpadDecimal": "NUMPAD_DEL",
    "NumpadDivide": "NUMPAD_DIVIDE",
    "NumpadMultiply": "NUMPAD_MULTIPLY",
    "NumpadSubtract": "NUMPAD_SUBTRACT",
    "NumpadAdd": "NUMPAD_ADD",
    "NumpadEnter": "NUMPAD_ENTER",
}


def _carb_name(browser: str) -> str | None:
    if browser in _SPECIAL:
        return _SPECIAL[browser]
    if browser.startswith("Key") and len(browser) == 4:
        return browser[3]
    if browser.startswith("Digit") and len(browser) == 6:
        return f"KEY_{browser[5]}"
    if browser.startswith("Numpad") and len(browser) == 7 and browser[6].isdigit():
        return f"NUMPAD_{browser[6]}"
    return None


def carb_to_vocabulary(vocabulary: Path) -> tuple[dict[str, str], list[str]]:
    """carb KeyboardInput name -> the controller's key name, and the untranslated."""
    keys = json.loads(vocabulary.read_text("utf-8"))["keys"]
    table: dict[str, str] = {}
    missing: list[str] = []
    for k in keys:
        carb = _carb_name(k["browser"])
        if carb is None:
            missing.append(k["name"])
        else:
            table[carb] = k["name"]
    return table, missing


class KeyEdges:
    """Key edges queued from the UI and handed over once a control tick.

    The same rule as `mjrl.app_play.ViewerKeys`: a key that went down this tick
    comes up at the next one at the earliest, because a press and a release
    handed over at one instant are never seen as held by a `toggle` or a `rise`.
    """

    def __init__(self, table: dict[str, str]) -> None:
        self._table = table
        self._queue: list[tuple[str, bool]] = []
        self._down: set[str] = set()
        self._lock = threading.Lock()

    def report(self, carb_name: str, down: bool) -> bool:
        """Queue an edge; returns whether the key is one the controller reads."""
        name = self._table.get(carb_name)
        if name is None:
            return False
        with self._lock:
            self._queue.append((name, down))
        return True

    def changes(self) -> tuple[list[str], list[str]]:
        with self._lock:
            queue, self._queue = self._queue, []
        downs: list[str] = []
        ups: list[str] = []
        later: list[tuple[str, bool]] = []
        for name, down in queue:
            if name in downs or any(n == name for n, _ in later):
                later.append((name, down))
            elif down and name not in self._down:
                self._down.add(name)
                downs.append(name)
            elif not down and name in self._down:
                self._down.discard(name)
                ups.append(name)
        with self._lock:
            self._queue[:0] = later
        return downs, ups

    def release_all(self) -> list[str]:
        """Every key the controller holds as down, forgotten (on reset)."""
        with self._lock:
            self._queue.clear()
        held, self._down = sorted(self._down), set()
        return held
