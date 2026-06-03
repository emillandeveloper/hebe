import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.debug import router
from app.services import db_sqlite


class DatabaseInspectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "hebe.db"
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                oauth_token TEXT,
                notes TEXT
            )
            """
        )
        conn.execute("CREATE TABLE empty_table (id INTEGER PRIMARY KEY, value TEXT)")
        conn.executemany(
            "INSERT INTO users (name, oauth_token, notes) VALUES (?, ?, ?)",
            [
                ("Leo", "abcd1234secretwxyz", "streamer"),
                ("Hebe", "short", "assistant"),
            ],
        )
        conn.commit()
        conn.close()

        app = FastAPI()
        app.state.adapter = SimpleNamespace(running=False, _engine=None)
        app.include_router(router)
        self.client = TestClient(app)
        self.patch = patch.object(db_sqlite, "DB_PATH", str(self.db_path))
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_list_tables_returns_counts(self):
        res = self.client.get("/debug/db/tables")

        self.assertEqual(res.status_code, 200)
        tables = {table["name"]: table for table in res.json()["tables"]}
        self.assertEqual(tables["users"]["row_count"], 2)
        self.assertEqual(tables["users"]["column_count"], 4)
        self.assertIn("empty_table", tables)

    def test_get_schema_returns_column_info(self):
        res = self.client.get("/debug/db/tables/users/schema")

        self.assertEqual(res.status_code, 200)
        columns = {column["name"]: column for column in res.json()["columns"]}
        self.assertTrue(columns["id"]["pk"])
        self.assertTrue(columns["oauth_token"]["sensitive"])

    def test_get_rows_supports_pagination(self):
        res = self.client.get("/debug/db/tables/users/rows?limit=1&offset=1")

        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["total"], 2)
        self.assertEqual(payload["limit"], 1)
        self.assertEqual(payload["offset"], 1)
        self.assertEqual(payload["rows"][0]["name"], "Hebe")

    def test_invalid_table_name_is_rejected(self):
        res = self.client.get("/debug/db/tables/missing_table/rows")

        self.assertEqual(res.status_code, 404)

    def test_sql_injection_like_table_name_is_rejected(self):
        res = self.client.get("/debug/db/tables/users%3BDROP%20TABLE%20users/rows")

        self.assertEqual(res.status_code, 404)
        with closing(sqlite3.connect(self.db_path)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        self.assertEqual(count, 2)

    def test_sensitive_columns_are_masked(self):
        res = self.client.get("/debug/db/tables/users/rows")

        self.assertEqual(res.status_code, 200)
        first = res.json()["rows"][0]
        second = res.json()["rows"][1]
        self.assertEqual(first["oauth_token"], "abcd********wxyz")
        self.assertEqual(second["oauth_token"], "[masked]")

    def test_missing_database_reports_not_found(self):
        self.patch.stop()
        with patch.object(db_sqlite, "DB_PATH", str(Path(self.tmp.name) / "missing.db")):
            res = self.client.get("/debug/db/tables")
        self.patch.start()

        self.assertEqual(res.status_code, 404)
        self.assertEqual(res.json()["detail"], "Database not found")


if __name__ == "__main__":
    unittest.main()
