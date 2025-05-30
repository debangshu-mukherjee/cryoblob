"""
Test runner script for local development.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    """Run the full test suite."""
    root_dir = Path(__file__).parent.parent
    success = True

    import os

    os.chdir(root_dir)

    test_commands = [
        (
            ["uv", "run", "black", "--check", "--diff", "src/", "tests/"],
            "Code formatting check",
        ),
        (["uv", "run", "pytest", "tests/", "-v", "--tb=short"], "Unit tests"),
        (
            [
                "uv",
                "run",
                "pytest",
                "tests/",
                "--cov=src/cryoblob",
                "--cov-report=term-missing",
            ],
            "Tests with coverage",
        ),
    ]

    for cmd, description in test_commands:
        if not run_command(cmd, description):
            success = False

    print(f"\n{'='*60}")
    if success:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
