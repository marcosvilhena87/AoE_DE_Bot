from dataclasses import dataclass, field
from typing import Any, MutableMapping, Iterator


@dataclass
class Config(MutableMapping[str, Any]):
    """Typed configuration container used across the project.

    It behaves like a mutable mapping so existing code expecting a dict-like
    object continues to work while providing a dedicated type that can be
    easily replaced in tests.
    """

    data: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:  # type: ignore[override]
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:  # type: ignore[override]
        self.data[key] = value

    def __delitem__(self, key: str) -> None:  # type: ignore[override]
        del self.data[key]

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        return iter(self.data)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.data)

    # Convenience helpers mirroring ``dict`` methods
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def update(self, *args, **kwargs) -> None:  # type: ignore[override]
        self.data.update(*args, **kwargs)

    def clear(self) -> None:  # type: ignore[override]
        self.data.clear()

    def copy(self) -> "Config":
        return Config(self.data.copy())
