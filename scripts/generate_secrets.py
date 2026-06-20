#!/usr/bin/env python3
"""Generate secure random secrets for the application.

Usage:
    python scripts/generate_secrets.py

Prints a fresh ADMIN_API_KEY (and any other secrets the stack needs) to stdout
so you can copy-paste them into .env.  Run this once before first deploy and
whenever you need to rotate credentials.
"""
from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_ENV_PATH = _ROOT / ".env"
_DEFAULT_VALUE = "change_me_before_deploy"


def _env_value(key: str) -> str | None:
    """Return the current value of *key* from .env, or None if not found."""
    if not _ENV_PATH.exists():
        return None
    for line in _ENV_PATH.read_text().splitlines():
        line = line.strip()
        if line.startswith(f"{key}="):
            return line[len(key) + 1:]
    return None


def main() -> None:
    admin_key = secrets.token_urlsafe(32)

    print("=" * 60)
    print("Generated secrets — copy these into your .env file")
    print("=" * 60)
    print()
    print(f"ADMIN_API_KEY={admin_key}")
    print()
    print("=" * 60)

    # Warn if the env file still has the placeholder value.
    current = _env_value("ADMIN_API_KEY")
    if current == _DEFAULT_VALUE:
        print(
            "\n⚠️  WARNING: .env still contains the default ADMIN_API_KEY.\n"
            "   Replace it with the value printed above before deploying.",
            file=sys.stderr,
        )
    elif current is None:
        print(
            "\nℹ️  .env not found or ADMIN_API_KEY not set.\n"
            "   Add the value printed above to your .env file.",
            file=sys.stderr,
        )
    else:
        print("\n✅ Reminder: update .env with the new value and restart the server.")


if __name__ == "__main__":
    main()
