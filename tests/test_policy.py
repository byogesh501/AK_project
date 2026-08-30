"""
Project Policy Enforcement Tests.
"""

import os
import re
from pathlib import Path

# Project root is the parent of the tests/ dir.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Patterns that load external pretrained weights/models --------------------
FORBIDDEN_PATTERNS = [
    re.compile(r'weights\s*='),
    re.compile(r'pretrained\s*='),
    re.compile(r'\.from_pretrained\('),
    re.compile(r'torch\.hub\.load\('),
]

# --- Directories explicitly excluded from scanning -----------------------------
SKIP_DIRS = {
    '.git', '.venv', 'venv', '__pycache__', '.pytest_cache',
    'data', 'logs', '.idea', '.vs',
}


def _iter_source_files(root: Path):
    """
    Yield every .py file under *root*, skipping generated / cache
    directories and the tests directory itself.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        # prune skipped dirs in-place so os.walk does not descend
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        # never scan inside the tests folder
        rel = Path(dirpath).relative_to(root)
        if rel.parts and rel.parts[0] == 'tests':
            continue

        for fname in filenames:
            if fname.endswith('.py'):
                yield Path(dirpath) / fname


def test_no_pretrained_weights_policy():
    """
    Enforces the strict project policy: inspection models MUST be trained
    from scratch. No external pretrained weights (e.g., ImageNet) are allowed.

    This automated guard scans every Python file under the project root
    (excluding tests, caches, and generated dirs) for forbidden syntax:
        weights=
        pretrained=
        .from_pretrained(
        torch.hub.load(

    Legitimate non-model uses can opt-out on a per-line basis with:
        # noqa: allow-pretrained
    """
    violations = []

    for py_file in _iter_source_files(PROJECT_ROOT):
        try:
            lines = py_file.read_text(encoding='utf-8').splitlines()
        except UnicodeDecodeError:
            # skip binary-ish or non-utf8 files
            continue

        for line_num, line in enumerate(lines, 1):
            # Skip comment-only lines
            if line.strip().startswith('#'):
                continue

            # Bypass switch for legitimate non-model uses
            if '# noqa: allow-pretrained' in line:
                continue

            for pattern in FORBIDDEN_PATTERNS:
                if pattern.search(line):
                    rel_path = py_file.relative_to(PROJECT_ROOT)
                    violations.append(
                        f"{rel_path}:{line_num} contains forbidden pattern '{pattern.pattern}'\n"
                        f"  > {line.strip()}"
                    )

    error_msg = (
        "Project Policy Violation: Pretrained models/weights are strictly forbidden.\n"
        "Inspection models must be trained from scratch. Found the following usages:\n\n"
        + "\n".join(violations)
        + "\n\nIf this is a legitimate non-model use, append `# noqa: allow-pretrained` to the line."
    )

    assert not violations, error_msg
