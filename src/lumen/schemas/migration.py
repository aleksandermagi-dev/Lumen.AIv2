from __future__ import annotations

from collections.abc import Callable
from typing import Any


MigrationFn = Callable[[dict[str, Any]], dict[str, Any]]


class SchemaMigration:
    """Applies explicit version upgrades for local persisted payloads."""

    def __init__(
        self,
        *,
        schema_type: str,
        current_version: str,
        migrations: dict[str, MigrationFn] | None = None,
        allow_newer_versions: bool = False,
    ) -> None:
        self.schema_type = schema_type
        self.current_version = str(current_version)
        self.migrations = dict(migrations or {})
        self.allow_newer_versions = bool(allow_newer_versions)

    def migrate(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        normalized.setdefault("schema_type", self.schema_type)
        version = str(normalized.get("schema_version") or "1")
        normalized["schema_version"] = version

        if normalized.get("schema_type") != self.schema_type:
            raise ValueError(
                f"Unsupported {self.schema_type} schema_type '{normalized.get('schema_type')}'"
            )

        if self.allow_newer_versions and self._is_newer_known_version(version):
            return normalized

        while version != self.current_version:
            migrate = self.migrations.get(version)
            if migrate is None:
                raise ValueError(
                    f"Unsupported {self.schema_type} schema_version '{version}'. "
                    f"Expected '{self.current_version}' or a known migration path."
                )
            normalized = migrate(dict(normalized))
            version = str(normalized.get("schema_version") or version)
            normalized["schema_version"] = version

        return normalized

    def _is_newer_known_version(self, version: str) -> bool:
        current = self._numeric_version(self.current_version)
        candidate = self._numeric_version(version)
        return current is not None and candidate is not None and candidate > current

    @staticmethod
    def _numeric_version(value: str) -> int | None:
        stripped = str(value or "").strip()
        if not stripped.isdigit():
            return None
        return int(stripped)
