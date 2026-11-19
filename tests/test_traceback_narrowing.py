"""DT-CORE-65: traceback frames narrow which class lost a private member.

Regression for issue #40: transformers 4.49 removed
``GenerationMixin._extract_past_from_model_output`` but the TensorFlow
``TFGenerationMixin`` still has it at 4.52.4. The old code abstained because
"some class still has it"; the traceback ran through generation.utils, the
module that defines GenerationMixin, not tf_utils.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from deptime.config import Coverage, PackageSpec, VersionPolicy
from deptime.index.store import IndexStore
from deptime.ingest.snapshot import build_snapshot
from deptime.query import DepTime

POLICY = VersionPolicy(
    since=date(2023, 1, 1), recent_months=12, include_prereleases=False, include_yanked=False
)
MEMBER = "_extract_past_from_model_output"


def _snapshots(tmp_path: Path, releases: dict) -> list:
    snaps = []
    for version, files in releases.items():
        root = tmp_path / version
        for rel, text in files.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(text)
        snaps.append(build_snapshot("gen", version, ["gen"], root, own_members=True))
    return snaps


def _index(tmp_path: Path) -> DepTime:
    releases = {
        "4.48.3": {
            "gen/__init__.py": (
                "from gen.generation.utils import GenerationMixin\n"
                "from gen.tf_utils import TFGenerationMixin\n"
            ),
            "gen/generation/__init__.py": "",
            "gen/generation/utils.py": (f"class GenerationMixin:\n    def {MEMBER}(self):\n        pass\n"),
            "gen/tf_utils.py": (f"class TFGenerationMixin:\n    def {MEMBER}(self):\n        pass\n"),
        },
        "4.52.4": {
            "gen/__init__.py": (
                "from gen.generation.utils import GenerationMixin\n"
                "from gen.tf_utils import TFGenerationMixin\n"
            ),
            "gen/generation/__init__.py": "",
            "gen/generation/utils.py": "class GenerationMixin:\n    pass\n",
            "gen/tf_utils.py": (f"class TFGenerationMixin:\n    def {MEMBER}(self):\n        pass\n"),
        },
    }
    spec = PackageSpec(name="gen", import_names=("gen",), tier="B")
    store = IndexStore.memory()
    store.rebuild_package(spec, _snapshots(tmp_path, releases))
    return DepTime(store, Coverage(POLICY, (spec,)))


def _tb(*frame_modules: str) -> str:
    lines = ["Traceback (most recent call last):"]
    lines.append('  File "/app/modeling.py", line 710, in _update')
    lines.append(f"    value = self.{MEMBER}(")
    for i, mod in enumerate(frame_modules):
        path = "/venv/lib/python3.12/site-packages/" + mod.replace(".", "/") + ".py"
        lines.append(f'  File "{path}", line {2597 + i}, in f{i}')
        lines.append("    value = next()")
    lines.append(f"AttributeError: 'CogVLMForCausalLM' object has no attribute '{MEMBER}'")
    return "\n".join(lines) + "\n"


def test_traceback_frame_narrows_which_class_lost_a_private_member(tmp_path):
    dt = _index(tmp_path)
    (f,) = dt.explain_error(_tb("gen.generation.utils"), {"gen": "4.52.4"}).findings
    assert (f.package, f.symbol, f.issue, f.last_version_with, f.confidence) == (
        "gen",
        f"gen.GenerationMixin.{MEMBER}",
        "symbol_removed",
        "4.48.3",
        "medium",
    )
    assert "ran through `gen.generation.utils`" in f.explanation


def test_no_frame_matching_a_candidate_module_abstains(tmp_path):
    dt = _index(tmp_path)
    assert dt.explain_error(_tb("other.mod"), {"gen": "4.52.4"}).findings == []


def test_more_than_one_candidate_remaining_abstains(tmp_path):
    dt = _index(tmp_path)
    assert dt.explain_error(_tb("gen.generation.utils", "gen.tf_utils"), {"gen": "4.52.4"}).findings == []


def test_no_frames_keeps_todays_answer(tmp_path):
    dt = _index(tmp_path)
    assert (
        dt.explain_error(
            f"AttributeError: 'CogVLMForCausalLM' object has no attribute '{MEMBER}'", {"gen": "4.52.4"}
        ).findings
        == []
    )


def _index_single_gone(tmp_path: Path) -> DepTime:
    """Only GenerationMixin ever had the member; no class has it at 4.52.4 (the DT-CORE-42 shape)."""
    releases = {
        "4.48.3": {
            "gen/__init__.py": "from gen.generation.utils import GenerationMixin\n",
            "gen/generation/__init__.py": "",
            "gen/generation/utils.py": f"class GenerationMixin:\n    def {MEMBER}(self):\n        pass\n",
        },
        "4.52.4": {
            "gen/__init__.py": "from gen.generation.utils import GenerationMixin\n",
            "gen/generation/__init__.py": "",
            "gen/generation/utils.py": "class GenerationMixin:\n    pass\n",
        },
    }
    spec = PackageSpec(name="gen", import_names=("gen",), tier="B")
    store = IndexStore.memory()
    store.rebuild_package(spec, _snapshots(tmp_path, releases))
    return DepTime(store, Coverage(POLICY, (spec,)))


def test_unmatched_frame_does_not_override_the_single_gone_class(tmp_path):
    """DT-CORE-42: an unrelated library frame must not erase the one-class answer (the gplearn case)."""
    dt = _index_single_gone(tmp_path)
    (f,) = dt.explain_error(_tb("gplearn.genetic"), {"gen": "4.52.4"}).findings
    assert (f.package, f.symbol, f.issue, f.last_version_with, f.confidence) == (
        "gen",
        f"gen.GenerationMixin.{MEMBER}",
        "symbol_removed",
        "4.48.3",
        "medium",
    )
    assert "only " in f.explanation
