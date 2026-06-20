"""CLI tool to create, list, and revoke Atticus access codes.

Usage:
  python scripts/manage_access_codes.py add --by "samarth" --note "Jones LLC" --uses 1
  python scripts/manage_access_codes.py add --code "MYCODE" --by "samarth"
  python scripts/manage_access_codes.py list
  python scripts/manage_access_codes.py revoke ATCX-7H2K-P9QR

Once any code exists in the database, new signups REQUIRE a valid code.
When the table is empty, signup is open to anyone (pilot mode).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dotenv import load_dotenv
load_dotenv()

from insightlens.config import ConfigError, load_config
from insightlens.storage.access_code_repository import AccessCodeRepository
from insightlens.storage.snowflake_client import SnowflakeConnectionError, open_connection


def cmd_add(args: argparse.Namespace) -> int:
    try:
        cfg = load_config()
        with open_connection(cfg.db) as conn:
            repo = AccessCodeRepository(conn)
            code = repo.create_code(
                created_by=args.by,
                max_uses=args.uses,
                note=args.note,
                code=args.code or None,
            )
        print(f"[access_codes] Created: {code}")
        if args.uses == 1:
            print(f"[access_codes] Single-use. Share this code with the invitee.")
        elif args.uses == 0:
            print(f"[access_codes] Unlimited-use code.")
        else:
            print(f"[access_codes] Max uses: {args.uses}")
    except (ConfigError, SnowflakeConnectionError) as exc:
        print(f"[access_codes] Error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    try:
        cfg = load_config()
        with open_connection(cfg.db) as conn:
            codes = AccessCodeRepository(conn).list_codes()
    except (ConfigError, SnowflakeConnectionError) as exc:
        print(f"[access_codes] Error: {exc}", file=sys.stderr)
        return 1

    if not codes:
        print("[access_codes] No codes found. Signup is currently open (no code required).")
        return 0

    print(f"{'CODE':<18} {'ACTIVE':<8} {'USES':<10} {'BY':<12} {'NOTE'}")
    print("-" * 70)
    for c in codes:
        uses = f"{c['uses_count']}/{c['max_uses'] or '∞'}"
        active = "YES" if c["is_active"] else "no"
        print(f"{c['code']:<18} {active:<8} {uses:<10} {c['created_by']:<12} {c['note'] or ''}")
    return 0


def cmd_revoke(args: argparse.Namespace) -> int:
    try:
        cfg = load_config()
        with open_connection(cfg.db) as conn:
            AccessCodeRepository(conn).revoke_code(args.code)
        print(f"[access_codes] Revoked: {args.code}")
    except (ConfigError, SnowflakeConnectionError) as exc:
        print(f"[access_codes] Error: {exc}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage Atticus access codes.")
    sub = parser.add_subparsers(dest="cmd")

    p_add = sub.add_parser("add", help="Create a new access code")
    p_add.add_argument("--by",   required=True, help="Your name / admin label")
    p_add.add_argument("--note", default=None,  help="Who this code is for")
    p_add.add_argument("--uses", type=int, default=1, help="Max uses (0=unlimited)")
    p_add.add_argument("--code", default=None, help="Custom code (auto-generated if omitted)")

    sub.add_parser("list", help="List all access codes")

    p_rev = sub.add_parser("revoke", help="Revoke an access code")
    p_rev.add_argument("code", help="The code to revoke")

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return 1

    dispatch = {"add": cmd_add, "list": cmd_list, "revoke": cmd_revoke}
    return dispatch[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
