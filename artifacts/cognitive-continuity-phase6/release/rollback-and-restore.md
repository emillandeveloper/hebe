# Phase 6 Backup, Rollback, and Restore

## Before production apply

1. Stop Hebe and confirm no SQLite writer remains.
2. Copy the configured DB, including `-wal`/`-shm` if present, or use SQLite's online backup API.
3. Record SHA-256 and run `PRAGMA integrity_check` on the backup.
4. Run `python -m app.integrity hygiene --db <copy> --dry-run` and `python -m app.integrity scan --strict` first.

## Restore

Stop Hebe, preserve the failed DB for forensic review, restore the verified backup to the configured path, then verify its SHA-256 and start Hebe offline before reconnecting external integrations.

## Downgrade boundary

Phase 6 schema changes are additive and old code ignores the audit tables. Code rollback is supported before safe hygiene apply. After any semantic invalidation/archive apply, restoring the pre-apply DB backup is the supported rollback; code rollback alone does not restore semantics. No physical delete/drop/purge exists in Phase 6.
