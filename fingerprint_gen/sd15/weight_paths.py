"""Default caption lexicon location."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEXICON = Path(os.environ.get("PROMPT_LEXICON", str(PROJECT_ROOT / "configs/livdet2015/prompt_lexicon.json"))).resolve()
