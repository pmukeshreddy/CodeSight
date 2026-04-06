from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def sample_repo(tmp_path: Path) -> Path:
    (tmp_path / "helpers.py").write_text(
        "\n".join(
            [
                "def normalize(value):",
                "    if value is None:",
                "        return ''",
                "    return value.strip()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        "\n".join(
            [
                "from helpers import normalize",
                "",
                "class Service:",
                "    def validate(self, payload):",
                "        return normalize(payload)",
                "",
                "def helper(value):",
                "    return normalize(value)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_app.py").write_text(
        "\n".join(
            [
                "from app import helper",
                "",
                "def test_helper():",
                "    assert helper(' x ') == 'x'",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return tmp_path
