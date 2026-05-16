"""Loader for the upstream persistent sheaf Laplacian implementation.

The original research implementation is not redistributed here. At publication
review time the upstream GitHub repository did not expose an explicit open-source
license, so this module loads a local copy supplied by the user instead.

Setup options, in order of precedence:
1. Set PSL_UPSTREAM_DIR to a directory containing PSL.py, or to the PSL.py file.
2. Clone https://github.com/weixiaoqimath/persistent_sheaf_Laplacians into
   external/persistent_sheaf_Laplacians/ at the repository root.
3. Put a legacy PSL.py file under analysis/ or scripts/.
"""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Iterable


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _candidate_paths() -> Iterable[Path]:
    env_value = os.environ.get("PSL_UPSTREAM_DIR")
    if env_value:
        env_path = Path(env_value).expanduser().resolve()
        yield env_path if env_path.name == "PSL.py" else env_path / "PSL.py"

    yield _REPO_ROOT / "external" / "persistent_sheaf_Laplacians" / "PSL.py"
    yield _REPO_ROOT / "third_party" / "persistent_sheaf_Laplacians" / "PSL.py"
    yield _REPO_ROOT / "analysis" / "PSL.py"
    yield _REPO_ROOT / "scripts" / "PSL.py"


@lru_cache(maxsize=1)
def _load_module() -> ModuleType:
    searched = []
    for candidate in _candidate_paths():
        searched.append(str(candidate))
        if not candidate.exists():
            continue
        spec = importlib.util.spec_from_file_location("_upstream_psl", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "PSL"):
            raise ImportError(f"Upstream PSL file found at {candidate}, but it does not define PSL.")
        return module

    message = "\n".join(f"  - {path}" for path in searched)
    raise ImportError(
        "Could not find the upstream persistent_sheaf_Laplacians PSL.py implementation.\n"
        "Clone https://github.com/weixiaoqimath/persistent_sheaf_Laplacians into "
        "external/persistent_sheaf_Laplacians/, or set PSL_UPSTREAM_DIR to the directory "
        "containing PSL.py. Searched:\n"
        f"{message}"
    )


class PSL:  # pragma: no cover - exercised only when the external PSL code is present
    """Proxy that instantiates the upstream :class:`PSL` class."""

    def __new__(cls, *args, **kwargs):
        upstream_cls = getattr(_load_module(), "PSL")
        return upstream_cls(*args, **kwargs)
