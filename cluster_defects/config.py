from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


class Config:
    def __init__(self, values: dict[str, Any], project_root: Path):
        self.values = values
        self.project_root = project_root.resolve()

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        config_path = Path(path).resolve()
        with config_path.open("rb") as handle:
            values = tomllib.load(handle)
        return cls(values, config_path.parent)

    def section(self, name: str) -> dict[str, Any]:
        return self.values[name]

    def get(self, section: str, key: str) -> Any:
        return self.values[section][key]

    def path(self, key: str, allow_empty: bool = False) -> Path | None:
        value = str(self.values["paths"].get(key, "")).strip()
        if not value:
            if allow_empty:
                return None
            raise ValueError(f"paths.{key} must not be empty")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    @property
    def output_root(self) -> Path:
        path = self.path("output_root")
        assert path is not None
        return path

