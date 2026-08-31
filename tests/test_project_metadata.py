from pathlib import Path
import re
import tomllib

from packaging.requirements import Requirement
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
FREEZE = ROOT / "requirements.freeze.txt"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"


def _metadata() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def _requirement_names(requirements: list[str]) -> set[str]:
    return {Requirement(requirement).name.lower() for requirement in requirements}


def test_project_metadata_describes_readme_and_supported_python() -> None:
    project = _metadata()["project"]

    assert project["readme"] == "README.md"
    assert project["requires-python"].startswith(">=3.11")
    assert "Programming Language :: Python :: 3.11" in project["classifiers"]
    assert "license" not in project
    assert "authors" not in project


def test_base_dependencies_cover_cpu_test_imports() -> None:
    project = _metadata()["project"]

    assert _requirement_names(project["dependencies"]) == {
        "numpy",
        "pillow",
        "pyyaml",
        "torch",
    }


def test_optional_dependency_groups_are_coherent() -> None:
    optional = _metadata()["project"]["optional-dependencies"]

    assert {"inference", "figures", "test", "full"} <= optional.keys()
    assert {"packaging", "pytest"} <= _requirement_names(optional["test"])
    assert "matplotlib" in _requirement_names(optional["figures"])
    assert "matplotlib" in _requirement_names(optional["full"])
    assert _requirement_names(optional["inference"]) <= _requirement_names(
        optional["full"]
    )


def test_full_cuda_freeze_is_exact_validated_direct_snapshot() -> None:
    expected = {
        "torch": "2.11.0+cu128",
        "torchvision": "0.26.0+cu128",
        "diffusers": "0.38.0",
        "transformers": "5.10.2",
        "accelerate": "1.13.0",
        "safetensors": "0.8.0rc1",
        "einops": "0.8.2",
        "pyyaml": "6.0.3",
        "pillow": "12.2.0",
        "numpy": "2.4.4",
        "tqdm": "4.68.1",
        "matplotlib": "3.11.0",
        "pytest": "9.0.3",
    }
    lines = [
        line.strip().lower()
        for line in FREEZE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    editable_lines = [line for line in lines if line.startswith("-e ")]
    requirement_lines = [line for line in lines if not line.startswith("-e ")]
    parsed = [Requirement(line) for line in requirement_lines]

    assert editable_lines == ["-e ."]
    assert len(requirement_lines) == len(expected)
    assert len({requirement.name.lower() for requirement in parsed}) == len(parsed)
    assert {
        requirement.name.lower(): str(requirement.specifier)
        for requirement in parsed
    } == {name: f"=={version}" for name, version in expected.items()}


def test_publication_metadata_has_no_placeholder_tokens_or_placeholder_locks() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (PYPROJECT, FREEZE)
    ).lower()

    assert not re.search(r"<(?:tested|replace|todo|version|commit)[^>]*>", text)
    assert not (ROOT / "requirements.lock.template").exists()
    assert not (ROOT / "uv.lock").exists()


def test_ci_is_private_safe_cpu_only_and_runs_full_suite() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    lower = text.lower()
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    job = workflow["jobs"]["cpu-unit-tests"]
    steps = job["steps"]
    install = next(step for step in steps if step.get("name") == "Install CPU test environment")
    run_tests = next(step for step in steps if step.get("name") == "Run full unit suite offline")

    assert workflow["permissions"] == {"contents": "read"}
    assert job["runs-on"] == "ubuntu-24.04"
    assert int(job["timeout-minutes"]) <= 20
    assert job["env"] == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DIFFUSERS_OFFLINE": "1",
    }
    assert steps[0]["uses"] == "actions/checkout@11d5960a326750d5838078e36cf38b85af677262"
    assert steps[1]["uses"] == "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065"
    assert steps[1]["with"]["python-version"] == "3.11"
    assert [line.strip() for line in install["run"].splitlines() if line.strip()] == [
        "python -m pip install pip==26.2.1 setuptools==78.1.0 wheel==0.46.3 packaging==26.0",
        "python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu",
        "python -m pip install numpy==2.4.4 Pillow==12.2.0 PyYAML==6.0.3 pytest==9.0.3",
        'python -m pip install -e ".[test]" --no-deps --no-build-isolation',
    ]
    assert run_tests["run"] == "python -m pytest -q"
    assert "cuda" not in lower
    assert "secrets." not in lower
