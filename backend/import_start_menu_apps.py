import os
import re
import sqlite3
from pathlib import Path
from datetime import datetime

import win32com.client  # pywin32


HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "hebe.db"  # ajusta si tu db está en otro sitio

START_DIRS = [
    Path(os.environ["APPDATA"]) / r"Microsoft\Windows\Start Menu\Programs",
    Path(os.environ["PROGRAMDATA"]) / r"Microsoft\Windows\Start Menu\Programs",
]

SKIP_WORDS = {
    "uninstall", "desinstal", "update", "updater", "actualiz",
    "help", "ayuda", "readme", "manual", "documentation", "docs",
    "license", "eula", "repair", "installer", "setup"
}


def norm_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[™®©]", "", s)
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def should_skip(shortcut_name: str) -> bool:
    n = norm_name(shortcut_name)
    return any(w in n for w in SKIP_WORDS)


def make_aliases(name_key: str, exe_name: str | None) -> str:
    # alias 1: acrónimo (visual studio code -> vsc)
    words = [w for w in re.split(r"\s+", name_key) if w]
    acro = "".join(w[0] for w in words if w and w[0].isalnum())
    aliases = set()
    if 3 <= len(acro) <= 8:
        aliases.add(acro)

    # alias 2: exe sin .exe
    if exe_name:
        aliases.add(exe_name.lower().removesuffix(".exe"))

    # alias 3: casos típicos
    if "visual studio code" in name_key:
        aliases.add("vscode")

    aliases.discard(name_key)
    aliases = [a for a in aliases if a]
    return ",".join(sorted(aliases))


def table_cols(cur) -> set[str]:
    return {r[1] for r in cur.execute("PRAGMA table_info(app_commands)").fetchall()}


def upsert_app(cur, *, name: str, command: str, description: str, aliases: str,
              process_name: str | None, window_title: str | None):
    cols = table_cols(cur)

    data = {
        "name": name,
        "command": command,
        "description": description,
        "aliases": aliases,
        "enabled": 1,
    }
    if "process_name" in cols:
        data["process_name"] = process_name
    if "window_title" in cols:
        data["window_title"] = window_title

    col_list = ", ".join(data.keys())
    ph_list = ", ".join(["?"] * len(data))

    # Estrategia:
    # - Si ya existe: NO machacar command salvo que el existente sea tipo "start xxx" o no tenga .exe.
    # - Sí rellenar process_name/window_title si están vacíos.
    cur.execute("SELECT * FROM app_commands WHERE name = ?", (name,))
    existing = cur.fetchone()

    if not existing:
        cur.execute(f"INSERT INTO app_commands ({col_list}) VALUES ({ph_list})", tuple(data.values()))
        return "insert"

    # update parcial
    updates = {}
    existing_cmd = (existing["command"] or "").strip().lower()

    better_cmd = (
        (".exe" in command.lower() or "shell:appsfolder" in command.lower())
        and (existing_cmd.startswith("start ") or ".exe" not in existing_cmd)
    )
    if better_cmd:
        updates["command"] = command

    # Rellenar huecos
    if "process_name" in cols and (not existing["process_name"]) and process_name:
        updates["process_name"] = process_name
    if "window_title" in cols and (not existing["window_title"]) and window_title:
        updates["window_title"] = window_title
    if (not existing["aliases"]) and aliases:
        updates["aliases"] = aliases
    if (not existing["description"]) and description:
        updates["description"] = description

    if not updates:
        return "skip"

    set_sql = ", ".join([f"{k} = ?" for k in updates.keys()])
    cur.execute(f"UPDATE app_commands SET {set_sql} WHERE name = ?", (*updates.values(), name))
    return "update"


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"No encuentro la DB en: {DB_PATH}")

    shell = win32com.client.Dispatch("WScript.Shell")

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # por si acaso (si no hiciste la migración en init_db)
    cols = table_cols(cur)
    if "process_name" not in cols:
        cur.execute("ALTER TABLE app_commands ADD COLUMN process_name TEXT")
    if "window_title" not in cols:
        cur.execute("ALTER TABLE app_commands ADD COLUMN window_title TEXT")

    inserted = updated = skipped = 0

    for base in START_DIRS:
        if not base.exists():
            continue

        for lnk in base.rglob("*.lnk"):
            shortcut_name = lnk.stem
            if should_skip(shortcut_name):
                continue

            try:
                sc = shell.CreateShortcut(str(lnk))
                target = getattr(sc, "TargetPath", "") or getattr(sc, "Targetpath", "") or ""
                args = getattr(sc, "Arguments", "") or ""
            except Exception:
                continue

            target = (target or "").strip()
            args = (args or "").strip()

            if not target:
                continue

            # UWP suele ser: explorer.exe + args "shell:AppsFolder\...."
            cmd = ""
            proc = None

            t_lower = target.lower()
            if t_lower.endswith("explorer.exe") and args.lower().startswith("shell:appsfolder"):
                cmd = f'explorer "{args}"'
                # process_name aquí es difícil (a veces ApplicationFrameHost.exe). Lo dejamos vacío.
            elif t_lower.endswith(".exe"):
                # cmd con comillas para rutas con espacios
                cmd = f'"{target}"' + (f" {args}" if args else "")
                proc = Path(target).name.lower()
            else:
                continue

            name_key = norm_name(shortcut_name)
            if not name_key:
                continue

            aliases = make_aliases(name_key, proc)
            desc = f"Importado de Inicio ({lnk.parent.name})"
            window_title = shortcut_name

            res = upsert_app(
                cur,
                name=name_key,
                command=cmd,
                description=desc,
                aliases=aliases,
                process_name=proc,
                window_title=window_title,
            )

            if res == "insert":
                inserted += 1
            elif res == "update":
                updated += 1
            else:
                skipped += 1

    conn.commit()
    conn.close()

    print(f"✅ Import fin: insert={inserted}, update={updated}, skip={skipped}")


if __name__ == "__main__":
    main()
