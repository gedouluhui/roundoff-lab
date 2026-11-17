"""Local-only file boundaries; no pickle, checkpoint execution, or networking."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import warnings

import numpy as np
from .quantize import Config


def load_npy(path: Path) -> np.ndarray:
    if path.suffix.lower() != ".npy":
        raise ValueError("inputs must be .npy files; export one float32/64 matrix per file")
    try:
        result = np.load(path, mmap_mode="r", allow_pickle=False, max_header_size=10_000)
    except (EOFError, ValueError) as exc:
        # NumPy 1.24 uses ValueError for an empty file; newer versions use EOFError.
        if isinstance(exc, EOFError) or path.stat().st_size == 0:
            raise ValueError("empty or truncated .npy file; export the matrix again") from exc
        raise
    if not isinstance(result, np.ndarray):
        if hasattr(result, "close"):
            result.close()
        raise ValueError("expected a single .npy matrix, not an archive")
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate configuration field: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"nonfinite JSON number is not allowed: {value}")


def load_configs(path: Path) -> list[Config]:
    if path.stat().st_size > 1_000_000:
        raise ValueError("configuration file exceeds 1 MB")
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                      parse_constant=_invalid_constant)
    if not isinstance(data, list) or not 1 <= len(data) <= 24:
        raise ValueError("configuration JSON must be a list of 1 to 24 objects")
    allowed = {"bits", "mode", "group_size", "percentile"}
    configs = []
    for entry in data:
        if not isinstance(entry, dict) or set(entry) - allowed:
            raise ValueError("configuration fields are bits, mode, group_size, percentile only")
        configs.append(Config(**entry))
    return configs


def npy_bytes(value: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, value, allow_pickle=False)
    return output.getvalue()


def write_artifacts(directory: Path, payloads: dict[str, bytes], *, force: bool = False,
                    inputs: tuple[Path, ...] = ()) -> None:
    """Preflight every destination, stage all bytes, then replace each file.

    Replacement is atomic per file, not a multi-file transaction. Do not run
    multiple writers against the same output directory.
    """
    if directory.is_symlink():
        raise ValueError("output directory must not be a symbolic link")
    targets = [directory / name for name in payloads]
    for target in targets:
        if target.is_symlink():
            raise ValueError("output artifacts must not be symbolic links")
        if target.exists() and not target.is_file():
            raise ValueError(f"output artifact is not a regular file: {target.name}")
        for source in inputs:
            if target.resolve() == source.resolve() or (
                    target.exists() and source.exists() and os.path.samefile(target, source)):
                raise ValueError("an output artifact aliases an input; choose another --out directory")
        if target.exists() and not force:
            raise ValueError(f"{target.name} already exists; choose a new --out directory or use --force")
    directory.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".roundoff-stage-", dir=directory))
    backups: list[tuple[Path, Path]] = []
    created: list[tuple[Path, Path]] = []
    retain_stage = False
    try:
        (stage / "new").mkdir()
        (stage / "old").mkdir()
        for name, content in payloads.items():
            (stage / "new" / name).write_bytes(content)
        # Move every old artifact aside before publishing any replacement.
        # In particular, a locked HTML file cannot leave a new JSON beside it.
        if force:
            for target in targets:
                if target.exists():
                    backup = stage / "old" / target.name
                    os.replace(target, backup)
                    backups.append((backup, target))
        for target in targets:
            source = stage / "new" / target.name
            # Atomic, exclusive publication on NTFS/POSIX filesystems. A writer
            # racing us cannot be silently overwritten. Unsupported filesystems
            # fail explicitly and trigger rollback.
            os.link(source, target)
            created.append((source, target))
    except BaseException as original:
        restore_errors = []
        for source, target in reversed(created):
            try:
                if target.exists():
                    if not os.path.samefile(source, target):
                        raise OSError("another process replaced an output during rollback")
                    target.unlink()
            except OSError as exc:
                restore_errors.append(str(exc))
        for backup, target in reversed(backups):
            try:
                if target.exists():
                    raise OSError("rollback destination changed; refusing to overwrite it")
                os.replace(backup, target)
            except OSError as exc:
                restore_errors.append(str(exc))
        if restore_errors:
            retain_stage = True
            raise OSError(f"output rollback incomplete; previous files retained in {stage / 'old'}") from original
        raise
    finally:
        if not retain_stage:
            try:
                shutil.rmtree(stage)
            except OSError:
                # Publication/rollback already finished. Garbage collection is
                # not part of the commit: leave the recoverable directory alone.
                warnings.warn(f"temporary-file cleanup incomplete; retained {stage}", RuntimeWarning, stacklevel=2)
