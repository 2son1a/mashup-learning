"""Delete directories matching glob patterns within the run directory.

Automatically preserves directories that are targets of symlinks (e.g.
the best-LR adapter selected by select_best).
"""

import shutil
from pathlib import Path

from omegaconf import DictConfig


def cleanup(cfg: DictConfig) -> None:
    """Remove directories matching glob patterns, respecting symlink targets.

    Config keys:
        output_dir          Used to derive the run directory (via parent).
        patterns            List of glob patterns to match for deletion.
        remove_symlinks     Optional list of glob patterns for symlinks to
                            remove before deletion.  Removing a symlink
                            unprotects its target so it can be deleted by
                            the subsequent patterns step.
    """
    output_dir = Path(cfg.output_dir)
    run_dir = output_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    patterns = list(cfg.get("patterns", []))
    if not patterns:
        print("Cleanup: no patterns specified, nothing to do")
        return

    remove_symlink_patterns = list(cfg.get("remove_symlinks", []))
    for pattern in remove_symlink_patterns:
        for match in sorted(run_dir.glob(pattern)):
            if match.is_symlink():
                match.unlink()
                print(f"  Removed symlink {match.name}")

    protected = set()
    for entry in run_dir.iterdir():
        if entry.is_symlink():
            target = (run_dir / entry.readlink()).resolve()
            protected.add(target)

    total_removed = 0
    total_bytes = 0

    for pattern in patterns:
        matches = sorted(run_dir.glob(pattern))
        for match in matches:
            if not match.is_dir() or match.is_symlink():
                continue
            if match.resolve() in protected:
                print(f"  Keeping {match.name} (symlink target)")
                continue
            size = sum(f.stat().st_size for f in match.rglob("*") if f.is_file())
            shutil.rmtree(match)
            total_removed += 1
            total_bytes += size

    gb = total_bytes / (1024**3)
    print(f"Cleanup: removed {total_removed} directories ({gb:.1f} GB freed)")
