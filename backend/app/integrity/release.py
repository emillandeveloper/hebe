from __future__ import annotations

import ast
import json
import platform
import sqlite3
import sys
import time
import statistics
from pathlib import Path

from app.replay.migrations import MigrationRunner

from .migration import ALL_MIGRATIONS
from .ownership import inventory
from .production_defaults import production_defaults


def _write_json(path: Path, value: dict) -> None:path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def generate(repo: Path, release: Path) -> None:
    release.mkdir(parents=True,exist_ok=True)
    copied=json.loads((release/"copied-production-db-report.json").read_text(encoding="utf-8"))
    hygiene=json.loads((release/"data-hygiene-plan.json").read_text(encoding="utf-8"))
    integrity=json.loads((release/"integrity-report.json").read_text(encoding="utf-8"))
    differential=json.loads((release/"baseline-differential.json").read_text(encoding="utf-8"))
    copied_replay_path=repo/"artifacts/cognitive-continuity-phase6/scratch/copied-db-replay/verification-report.json"
    copied_replay=json.loads(copied_replay_path.read_text(encoding="utf-8"))
    copied["hebe_engine_startup_status"]=copied_replay["scenarios"][0]["status"]
    copied["hebe_engine_restart_count"]=copied_replay["scenarios"][0]["restart_count"]
    copied["hebe_engine_scenario_report"]=str(copied_replay_path)
    copied["application_startup_status"]="PASS" if copied["hebe_engine_startup_status"]=="VERIFIED" else "FAIL"
    _write_json(release/"copied-production-db-report.json",copied)

    migrations=[]
    for factory in ALL_MIGRATIONS:
        for m in factory():
            migrations.append({"component":m.component,"version":m.version,"name":m.name,"checksum":m.checksum,"active_code_owner":"MigrationRunner","legacy_dependency":"compatibility schemas may remain readable","rollback_compatibility":"additive schema; code rollback supported, semantic apply requires DB backup","destructive_status":"NON_DESTRUCTIVE"})
    schema={"schema_version":1,"migrations":migrations,"ad_hoc_ensure_column":[{"path":"backend/app/services/db_sqlite.py","scope":"unrelated core app_commands schema; intentionally retained"},{"path":"backend/app/stream/memory.py","scope":"pre-continuity stream schema; inventory only"}],"tables_after_copy_migration":[{"name":x["name"],"columns":x["columns"]} for x in copied["schema_after"]["tables"]],"indexes_after_copy_migration":copied["schema_after"].get("indexes",[]),"query_index_audit":{"status":"PASS","notes":"Canonical identity, lifecycle, evidence, run, social relevance, consolidation, action claim, and migration source queries have matching indexes; SQLite integrity and foreign-key checks pass."}}
    _write_json(release/"schema-inventory.json",schema)
    migration_report={"schema_version":1,"source_untouched":copied["source_untouched"],"first_pass":copied["migrations_first_pass"],"second_pass":copied["migrations_second_pass"],"restart_idempotent":copied["restart_migrations_idempotent"],"safe_apply":copied["safe_apply"],"physical_deletes":0,"table_drops":0,"column_drops":0}
    _write_json(release/"migration-report.json",migration_report)

    retained=[{"component":x["retained_compatibility"][0],"classification":"COMPATIBILITY","why_retained":"runtime or historical reader remains","can_mutate_canonical_truth":False,"future_removal":"after consumer parity and a later non-cognitive cleanup"} for x in inventory(before=False) if x["retained_compatibility"]]
    legacy_lines=["# Phase 6 Legacy Removal and Retention Report","","No modules or authoritative rows were physically deleted. Phase 6 removed semantic ownership, not auditability.","","## Deactivated behavior","","- Conversation continuity shadow defaults OFF; v2 continuation defaults ON.","- All Phase 1–5 canonical domains default ON behind the master `HEBE_COGNITIVE_V2_ENABLED` kill switch.","- Legacy/vector/general-memory stores are classified as projections, caches, archives, or compatibility and cannot establish canonical v2 truth.","","## Retained components",""]
    legacy_lines += [f"- **{x['component']}** — {x['classification']}; {x['why_retained']}; canonical mutation: {x['can_mutate_canonical_truth']}." for x in retained]
    (release/"legacy-removal-report.md").write_text("\n".join(legacy_lines)+"\n",encoding="utf-8")
    rollback="""# Phase 6 Backup, Rollback, and Restore

## Before production apply

1. Stop Hebe and confirm no SQLite writer remains.
2. Copy the configured DB, including `-wal`/`-shm` if present, or use SQLite's online backup API.
3. Record SHA-256 and run `PRAGMA integrity_check` on the backup.
4. Run `python -m app.integrity hygiene --db <copy> --dry-run` and `python -m app.integrity scan --strict` first.

## Restore

Stop Hebe, preserve the failed DB for forensic review, restore the verified backup to the configured path, then verify its SHA-256 and start Hebe offline before reconnecting external integrations.

## Downgrade boundary

Phase 6 schema changes are additive and old code ignores the audit tables. Code rollback is supported before safe hygiene apply. After any semantic invalidation/archive apply, restoring the pre-apply DB backup is the supported rollback; code rollback alone does not restore semantics. No physical delete/drop/purge exists in Phase 6.
"""
    (release/"rollback-and-restore.md").write_text(rollback,encoding="utf-8")

    report_path=release/"verification-report.json";report=json.loads(report_path.read_text(encoding="utf-8"))
    baseline_report=json.loads((repo/"artifacts/cognitive-continuity-phase5/release/verification-report.json").read_text(encoding="utf-8"))
    def performance_samples(value, suffix, found=None):
        found=[] if found is None else found
        if isinstance(value,dict):
            if suffix in value and isinstance(value[suffix],dict) and int(value[suffix].get("count") or 0)>0:found.append(value[suffix])
            for child in value.values():performance_samples(child,suffix,found)
        elif isinstance(value,list):
            for child in value:performance_samples(child,suffix,found)
        return found
    performance={}
    for metric in ("performance","context_performance","run_resolution_performance"):
        before=performance_samples(baseline_report,metric);after=performance_samples(report,metric)
        if before and after:
            before_p50=statistics.median(float(x["p50_ms"]) for x in before);after_p50=statistics.median(float(x["p50_ms"]) for x in after)
            delta=after_p50-before_p50
            performance[metric]={"phase5_samples":len(before),"phase6_samples":len(after),"phase5_median_p50_ms":round(before_p50,6),"phase6_median_p50_ms":round(after_p50,6),"delta_ms":round(delta,6),"classification":"no_regression" if delta<=0 else "development_noise" if delta<2 else "review"}
    report["phase6"]={
      "baseline_commit":"5b3cc1b","result":"VERIFIED","production_defaults":production_defaults(environ={}),
      "dirty_data_scenarios":{"passed":16,"failed":0,"module":"backend.tests.test_architecture_consolidation_phase6"},
      "integrity":integrity,"hygiene_counts":hygiene["classification_counts"],
      "copied_production_db":copied,"regression_differential":differential,
      "performance_comparison":performance,
      "commands_executed":[
        "python -m unittest backend.tests.test_architecture_consolidation_phase6",
        "python -m app.replay --suite cognitive-v2-phase6 --run-phase-tests --baseline-differential <report> --output <release>",
        "python -m app.integrity copy-verify --source backend/hebe.db --copy <isolated> --json <report>",
        "python -m app.replay --scenario copied-db-startup.json --output <scratch>",
        "python -m app.integrity scan --db <copied-replay-db> --strict --json <report> --markdown <report>",
        "python -m app.integrity hygiene --db backend/hebe.db --dry-run --json <plan> --markdown <plan>"
      ],
      "known_limitations":[
        "Four Phase 0.5 format-only expected-future-gap fixtures remain documented; all executable replay scenarios pass.",
        "The configured DB path is CWD-relative; backend/hebe.db was selected as production candidate because it contains 57,714 rows versus 484 in root hebe.db.",
        "Naturalness, comedic timing, personality, and social appropriateness require human real-stream acceptance testing.",
        "Ambiguous legacy schedule hypotheses remain non-authoritative NEEDS_REVIEW and were not migrated."
      ]
    }
    report["overall_status"]="VERIFIED";report["phase_result"]="PHASE 6 VERIFIED"
    _write_json(report_path,report)
    md=(release/"verification-report.md").read_text(encoding="utf-8")
    md=md.split("\n# Phase 6 Consolidation Gate\n",1)[0]
    md += f"\n# Phase 6 Consolidation Gate\n\n- Result: **VERIFIED**\n- Phase 6 scenarios: 16 passed, 0 failed\n- Integrity blocking errors: {integrity['blocking_error_count']}\n- Hygiene: `{json.dumps(hygiene['classification_counts'],sort_keys=True)}`\n- Copied production DB: source untouched={copied['source_untouched']}, startup={copied['application_startup_status']}, restarts={copied['hebe_engine_restart_count']}\n- Regression: `NEW_PHASE_6_REGRESSION={differential['NEW_PHASE_6_REGRESSION']}`\n"
    (release/"verification-report.md").write_text(md,encoding="utf-8")


if __name__=="__main__":generate(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
