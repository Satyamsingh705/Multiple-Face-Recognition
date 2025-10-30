import argparse
import os
import sqlite3
from typing import Optional


def ensure_persons_pk(db_path: str) -> None:
    """
    Ensure 'persons.id' is PRIMARY KEY (unique, non-NULL).
    If not, migrate the table schema atomically.
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()

        # Ensure table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='persons'")
        if cur.fetchone() is None:
            raise RuntimeError("Table 'persons' not found in database")

        # If 'id' already a PRIMARY KEY, nothing to do
        cur.execute("PRAGMA table_info(persons)")
        cols = cur.fetchall()  # (cid, name, type, notnull, dflt_value, pk)
        id_is_pk = any(name == "id" and int(pk) > 0 for _, name, _, _, _, pk in cols)
        if id_is_pk:
            return

        # Validate no NULL ids and no duplicates before migration
        cur.execute("SELECT COUNT(*) FROM persons WHERE id IS NULL")
        null_count = cur.fetchone()[0]
        if null_count:
            raise RuntimeError("Cannot set 'id' as PRIMARY KEY: found NULL id(s). Fix IDs first.")

        cur.execute("SELECT id, COUNT(*) FROM persons GROUP BY id HAVING COUNT(*) > 1")
        dup_rows = cur.fetchall()
        if dup_rows:
            dup_ids = [row[0] for row in dup_rows]
            raise RuntimeError(f"Cannot set 'id' as PRIMARY KEY: duplicate id(s) found {dup_ids}. Use different IDs.")

        # Migrate schema to enforce PRIMARY KEY(id)
        conn.execute("BEGIN IMMEDIATE")
        cur.execute("""
            CREATE TABLE persons__new (
                id INTEGER PRIMARY KEY NOT NULL,
                name TEXT UNIQUE,
                embedding BLOB
            )
        """)
        cur.execute("INSERT INTO persons__new (id, name, embedding) SELECT id, name, embedding FROM persons")
        cur.execute("DROP TABLE persons")
        cur.execute("ALTER TABLE persons__new RENAME TO persons")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(description="Ensure persons.id is PRIMARY KEY in embeddings.db")
    default_db = os.path.join(os.path.dirname(__file__), "embeddings.db")
    ap.add_argument("--db", default=default_db, help="Path to embeddings.db (default: embeddings.db next to this script)")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        raise FileNotFoundError(f"Database not found: {args.db}. Pass --db with the correct path.")

    ensure_persons_pk(args.db)
    print("persons.id is enforced as PRIMARY KEY.")
    

if __name__ == "__main__":
    main()
