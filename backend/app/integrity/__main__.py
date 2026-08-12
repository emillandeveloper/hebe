from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .hygiene import HygienePlanner
from .ownership import inventory
from .scanner import IntegrityScanner
from .migration import verify_copied_database, write_report


def _write(path: str | None, content: str) -> None:
    if not path:return
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(content,encoding="utf-8")


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(prog="python -m app.integrity")
    sub=parser.add_subparsers(dest="command",required=True)
    scan=sub.add_parser("scan");scan.add_argument("--db",required=True);scan.add_argument("--json");scan.add_argument("--markdown");scan.add_argument("--strict",action="store_true")
    hygiene=sub.add_parser("hygiene");hygiene.add_argument("--db",required=True);hygiene.add_argument("--json");hygiene.add_argument("--markdown");m=hygiene.add_mutually_exclusive_group();m.add_argument("--dry-run",action="store_true");m.add_argument("--apply-safe",action="store_true")
    own=sub.add_parser("ownership");own.add_argument("--before",action="store_true");own.add_argument("--after",action="store_true");own.add_argument("--json",required=True)
    copied=sub.add_parser("copy-verify");copied.add_argument("--source",required=True);copied.add_argument("--copy",required=True);copied.add_argument("--json",required=True);copied.add_argument("--no-apply-safe",action="store_true")
    args=parser.parse_args(argv)
    if args.command=="scan":
        report=IntegrityScanner(args.db).scan();_write(args.json,json.dumps(report,ensure_ascii=False,indent=2)+"\n");_write(args.markdown,IntegrityScanner.markdown(report));print(json.dumps({"status":report["status"],"blocking_errors":report["blocking_error_count"],"counts":report["counts"]}));return 2 if args.strict and report["blocking_error_count"] else 0
    if args.command=="hygiene":
        planner=HygienePlanner(args.db);report=planner.plan()
        if args.apply_safe:report=planner.apply_safe(report)
        _write(args.json,json.dumps(report,ensure_ascii=False,indent=2)+"\n");_write(args.markdown,HygienePlanner.markdown(report));print(json.dumps({"mode":report["mode"],"counts":report["classification_counts"],"destructive_changes":report["destructive_changes"]}));return 0
    if args.command=="copy-verify":
        report=verify_copied_database(args.source,args.copy,apply_safe=not args.no_apply_safe);write_report(report,args.json);print(json.dumps({"source_untouched":report["source_untouched"],"integrity":report["integrity_status"],"startup":report["application_startup_status"]}));return 0 if report["source_untouched"] and report["integrity_status"]=="PASS" and report["application_startup_status"]=="PASS" else 2
    before=args.before or not args.after;report={"schema_version":1,"stage":"before" if before else "after","concerns":inventory(before=before)};_write(args.json,json.dumps(report,ensure_ascii=False,indent=2)+"\n");return 0


if __name__=="__main__":raise SystemExit(main())
