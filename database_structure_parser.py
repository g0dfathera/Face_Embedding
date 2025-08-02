import sqlite3

DB_PATH = "path/to/your/database.db"

def inspect_database_schema():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("\n Tables and their columns:\n" + "-"*50)

    # Fetch all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for (table_name,) in tables:
        print(f"\n Table: {table_name}")
        try:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            for col in columns:
                col_id, name, dtype, notnull, default, pk = col
                print(f"  - {name} ({dtype})")
        except Exception as e:
            print(f"  [!] Error reading table '{table_name}': {e}")

    cursor.close()
    conn.close()
    print("\n Done inspecting schema.")

if __name__ == "__main__":
    inspect_database_schema()
