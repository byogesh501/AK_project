"""
Project Policy Enforcement Tests.
"""

import os
import re
from pathlib import Path

# Project root is the parent.parent of this file since this is tests/test_policy.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_PATTERNS = [
    re.compile(r'weights\s*='),
    re.compile(r'pretrained\s*='),
    re.compile(r'\.from_pretrained\('),
    re.compile(r'torch\.hub\.load\('),
]

def test_no_pretrained_weights_policy():
    """
    Enforces the strict project policy: Core inspection models MUST be trained
    from scratch. No external pretrained weights (e.g., ImageNet) are allowed.

    This automated guard scans all `.py` files in `core/` and `domains/`
    for forbidden syntax like `weights=`, `pretrained=`, etc.
    """
    violations = []
    dirs_to_check = ['core', 'domains']

    for d in dirs_to_check:
        dir_path = PROJECT_ROOT / d
        if not dir_path.exists():
            continue

        for root, _, files in os.walk(dir_path):
            for file in files:
                if not file.endswith('.py'):
                    continue

                file_path = Path(root) / file

                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines, 1):
                    # Skip comment lines
                    if line.strip().startswith('#'):
                        continue

                    # Legitimate non-model bypass (e.g. string matching logic not related to model loading)
                    if 'noqa: allow-pretrained' in line:
                        continue

                    for pattern in FORBIDDEN_PATTERNS:
                        if pattern.search(line):
                            rel_path = file_path.relative_to(PROJECT_ROOT)
                            violations.append(
                                f"{rel_path}:{line_num} contains forbidden pattern '{pattern.pattern}'\n"
                                f"  > {line.strip()}"
                            )

    error_msg = (
        "Project Policy Violation: Pretrained models/weights are strictly forbidden.\n"
        "Models must be trained from scratch. Found the following usages:\n\n" +
        "\n".join(violations) +
        "\n\nIf this is a legitimate non-model use, append `# noqa: allow-pretrained` to the line."
    )

    assert not violations, error_msg
