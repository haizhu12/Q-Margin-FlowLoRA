from pathlib import Path
import re
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PUBLISHABLE_TEXT_SUFFIXES = {".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
PUBLISHABLE_ROOT_FILES = {
    "README.md",
    "pyproject.toml",
    "requirements.freeze.txt",
    "requirements.lock.template",
}
PUBLISHABLE_EXPLICIT_FILES = (
    "data/README.md",
    "data/subjects/subjects_sop_small_valid.jsonl",
    "docs/REPRODUCIBILITY.md",
)
PUBLISHABLE_DATA_GLOBS = ("data/eval/*.jsonl",)
PUBLISHABLE_SOURCE_DIRS = (
    ".github",
    "configs",
    "demos",
    "qmargin",
    "scripts",
    "tests",
)
PUBLISHABLE_CACHE_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
MACHINE_SPECIFIC_WINDOWS_PATH = re.compile(
    r"(?:f:(?:\\+|/+)phd_work|c:(?:\\+|/+)users(?:\\+|/))",
    flags=re.IGNORECASE,
)


def _publishable_text_files() -> list[Path]:
    files = [
        ROOT / name for name in PUBLISHABLE_ROOT_FILES if (ROOT / name).is_file()
    ]
    files.extend(
        ROOT / relative_path
        for relative_path in PUBLISHABLE_EXPLICIT_FILES
        if (ROOT / relative_path).is_file()
    )
    for pattern in PUBLISHABLE_DATA_GLOBS:
        files.extend(path for path in ROOT.glob(pattern) if path.is_file())
    for dirname in PUBLISHABLE_SOURCE_DIRS:
        files.extend(
            path
            for path in (ROOT / dirname).rglob("*")
            if (
                path.is_file()
                and path.suffix.lower() in PUBLISHABLE_TEXT_SUFFIXES
                and not PUBLISHABLE_CACHE_DIRS.intersection(path.relative_to(ROOT).parts)
            )
        )
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def _contains_machine_specific_windows_path(text: str) -> bool:
    return MACHINE_SPECIFIC_WINDOWS_PATH.search(text) is not None


def _is_ignored(repo: Path, relative_path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", relative_path],
        cwd=repo,
        check=False,
    )
    assert result.returncode in {0, 1}, f"git check-ignore failed for {relative_path}"
    return result.returncode == 0


def test_publishable_source_has_no_machine_specific_windows_paths() -> None:
    offenders = []
    for path in _publishable_text_files():
        text = path.read_text(encoding="utf-8")
        if _contains_machine_specific_windows_path(text):
            offenders.append(path.relative_to(ROOT).as_posix())

    assert sorted(offenders) == [], f"machine-specific paths found in: {sorted(offenders)}"


def test_publishable_text_files_skip_missing_candidates_and_are_sorted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys.modules[__name__],
        "PUBLISHABLE_ROOT_FILES",
        {"requirements.freeze.txt", "not-present.txt", "pyproject.toml"},
    )
    monkeypatch.setattr(sys.modules[__name__], "PUBLISHABLE_EXPLICIT_FILES", ())
    monkeypatch.setattr(sys.modules[__name__], "PUBLISHABLE_DATA_GLOBS", ())
    monkeypatch.setattr(sys.modules[__name__], "PUBLISHABLE_SOURCE_DIRS", ())

    assert [path.name for path in _publishable_text_files()] == [
        "pyproject.toml",
        "requirements.freeze.txt",
    ]


def test_publishable_discovery_includes_only_version_eligible_data_files() -> None:
    expected = {
        ROOT / "data" / "README.md",
        ROOT / "data" / "subjects" / "subjects_sop_small_valid.jsonl",
        *(ROOT / "data" / "eval").glob("*.jsonl"),
    }
    discovered = {
        path for path in _publishable_text_files() if path.is_relative_to(ROOT / "data")
    }

    assert discovered == expected


@pytest.mark.parametrize(
    "text",
    [
        "F:" + "\\Phd_Work\\project",
        "F:" + "/Phd_Work/project",
        "F:" + "\\\\Phd_Work\\\\project",
        "C:" + "\\Users\\researcher\\project",
        "C:" + "/Users/researcher/project",
        "C:" + "\\\\Users\\\\researcher\\\\project",
    ],
)
def test_machine_specific_path_detection_catches_common_source_spellings(
    text: str,
) -> None:
    assert _contains_machine_specific_windows_path(text)


@pytest.mark.parametrize("text", ["/mnt/data/project", "data/eval/file.jsonl"])
def test_machine_specific_path_detection_accepts_portable_paths(text: str) -> None:
    assert not _contains_machine_specific_windows_path(text)


def test_gitignore_enforces_publication_allowlist(tmp_path: Path) -> None:
    repo = tmp_path / "ignore-semantics"
    repo.mkdir()
    (repo / ".gitignore").write_text(
        (ROOT / ".gitignore").read_text(encoding="utf-8"), encoding="utf-8"
    )
    subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)

    ignored = {
        "models/model.safetensors",
        "outputs/run/result.json",
        "tmp/scratch.txt",
        "qmargin/__pycache__/module.cpython-311.pyc",
        ".pytest_tmp/session/state",
        "data/raw/sop/archive.zip",
        "data/eval/metrics.csv",
        "data/eval/nested/hidden.jsonl",
        "data/subjects/subjects_sop_smoke.jsonl",
        "data/subjects/subjects_sop_small_valid_summary.json",
        "docs/internal-notes.md",
        "docs/superpowers/plans/private-plan.md",
        ".vscode/settings.json",
        ".idea/workspace.xml",
        ".env",
        ".venv/pyvenv.cfg",
        "run.log",
        "build/package/file.py",
        "dist/package.whl",
        ".DS_Store",
        "Thumbs.db",
    }
    publishable = {
        "data/README.md",
        "data/eval/benchmark.jsonl",
        "data/subjects/subjects_sop_small_valid.jsonl",
        "docs/REPRODUCIBILITY.md",
        "demos/outputs/example.txt",
        "qmargin/inference_runtime.py",
        "qmargin/models/component.py",
    }

    not_ignored = sorted(path for path in ignored if not _is_ignored(repo, path))
    unexpectedly_ignored = sorted(
        path for path in publishable if _is_ignored(repo, path)
    )

    assert not_ignored == []
    assert unexpectedly_ignored == []
