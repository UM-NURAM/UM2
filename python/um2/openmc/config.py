"""Global configuration (compatible with :data:`openmc.config`)."""

from __future__ import annotations

from collections.abc import MutableMapping
from contextlib import contextmanager
import os
from pathlib import Path
import warnings
from typing import Any, Iterator

__all__ = ["config"]


class _Config(MutableMapping):
    """Mapping of configuration keys to values.

    Path-valued keys are mirrored into the environment variables used by
    OpenMC, so that ``os.environ['OPENMC_MG_CROSS_SECTIONS'] = ...`` and
    ``config['mg_cross_sections'] = ...`` are interchangeable.
    """

    _PATH_KEYS: dict[str, str] = {
        'cross_sections': 'OPENMC_CROSS_SECTIONS',
        'mg_cross_sections': 'OPENMC_MG_CROSS_SECTIONS',
        'chain_file': 'OPENMC_CHAIN_FILE',
    }

    def __init__(self, data: dict = ()):
        self._mapping: dict[str, Any] = {'resolve_paths': True}
        self.update(data)

    def __getitem__(self, key: str) -> Any:
        return self._mapping[key]

    def __delitem__(self, key: str):
        if key == 'resolve_paths':
            raise KeyError("'resolve_paths' cannot be deleted.")
        del self._mapping[key]
        env_var = self._PATH_KEYS.get(key)
        if env_var in os.environ:
            del os.environ[env_var]

    def __setitem__(self, key: str, value: Any):
        if key in self._PATH_KEYS:
            p = Path(value)
            stored_path = p.resolve(strict=False) if self._mapping.get(
                'resolve_paths', True) else p
            self._mapping[key] = stored_path
            os.environ[self._PATH_KEYS[key]] = str(stored_path)
            if not stored_path.exists():
                warnings.warn(f"Path '{stored_path}' does not exist.", UserWarning)
        elif key == 'resolve_paths':
            if not isinstance(value, bool):
                raise TypeError("'resolve_paths' must be a boolean.")
            self._mapping[key] = value
        else:
            valid_keys = list(self._PATH_KEYS) + ['resolve_paths']
            raise KeyError(f"Unrecognized config key: {key}. Acceptable keys are: "
                           f"{', '.join(repr(k) for k in valid_keys)}.")

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def __repr__(self) -> str:
        return repr(self._mapping)

    def clear(self):
        for key in [k for k in self._mapping if k != 'resolve_paths']:
            del self[key]

    @contextmanager
    def patch(self, key: str, value: Any):
        """Temporarily change a value in the configuration."""
        previous_value = self.get(key)
        self[key] = value
        try:
            yield
        finally:
            if previous_value is None:
                del self[key]
            else:
                self[key] = previous_value

    def get_path(self, key: str):
        """Return the path for a key, also checking the environment variable,
        since scripts commonly set ``os.environ[...]`` directly."""
        env_var = self._PATH_KEYS[key]
        if env_var in os.environ:
            value = Path(os.environ[env_var])
            if self._mapping.get(key) != value:
                self._mapping[key] = value
        return self._mapping.get(key)


def _default_config(**kwargs) -> _Config:
    config = _Config(kwargs)
    for key, var in _Config._PATH_KEYS.items():
        if var in os.environ:
            config[key] = os.environ[var]
    return config


#: Global configuration dictionary
config = _default_config()
