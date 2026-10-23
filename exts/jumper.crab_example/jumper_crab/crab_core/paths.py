"""Where the crab example finds its inputs.

Everything is resolved from this file's location in the repository, so the
example finds the same files whether Isaac Sim loads it as an extension or a
tool imports it:

    <repo>/data/crab.usd                                  the robot, tools/build_usd.py
    <repo>/data/crab_model.json                           the training model, tools/export_model.py
    <repo>/third_party/jumper/out/bundle_example/jumper   the app: controller, policies, contracts
    <repo>/third_party/jumper/controller/vocabulary.json  key names the controller understands

Overrides: `CRAB_DATA` (the generated files), `CRAB_JUMPER_ROOT` (a Jumper
checkout elsewhere), `CRAB_BUNDLE` (a different bundle).
"""

from __future__ import annotations

import os
from pathlib import Path

#: <repo>/exts/jumper.crab_example/jumper_crab/crab_core/paths.py -> <repo>
REPO = Path(__file__).resolve().parents[4]
DATA = Path(os.environ.get("CRAB_DATA", REPO / "data"))
USD_PATH = DATA / "crab.usd"
MODEL_JSON = DATA / "crab_model.json"

JUMPER_ROOT = Path(os.environ.get("CRAB_JUMPER_ROOT", REPO / "third_party" / "jumper"))
BUNDLE = Path(os.environ.get("CRAB_BUNDLE", JUMPER_ROOT / "out" / "bundle_example" / "jumper"))
VOCABULARY = JUMPER_ROOT / "controller" / "vocabulary.json"
