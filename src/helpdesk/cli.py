import argparse
import sys
from pathlib import Path

from .part1.cli import PREDICTIONS, run_part1_cli


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="HW5 helpdesk: original/main (Part 1) or agent (Part 2)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--message")
    source.add_argument("--scenario", choices=(*PREDICTIONS, "S06", "S07", "S08", "S09", "S14"))
    source.add_argument("--interactive", action="store_true")
    parser.add_argument("--graph", choices=("original", "main", "agent"), default=None,
                        help="Default: main for messages; required fixture variant for scenarios")
    parser.add_argument("--output", type=Path, help="Save the complete JSON trace")
    parser.add_argument("--user-id", default="demo-user-a")
    parser.add_argument("--thread-id", default="demo-thread")
    parser.add_argument("--status-profile", choices=("normal", "wifi_outage", "recovery_outage", "status_unavailable"), default="normal")
    parser.add_argument("--storage-dir", type=Path)
    args = parser.parse_args(argv)
    if args.interactive and args.graph != "agent":
        parser.error("--interactive requires --graph agent")
    if args.graph == "agent" or (args.scenario and args.scenario not in PREDICTIONS):
        if args.scenario and args.graph and args.graph != "agent":
            parser.error(f"{args.scenario} requires --graph agent")
        if args.scenario in PREDICTIONS:
            parser.error("Part 1 scenarios require their original/main graph")
        from .part2.cli import run_agent_cli
        return run_agent_cli(args)
    return run_part1_cli(args, sys.argv[1:] if argv is None else argv, parser)


if __name__ == "__main__":
    # Model replies may contain characters outside Windows' legacy code page.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
