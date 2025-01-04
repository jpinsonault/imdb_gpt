import sqlite3

def print_schema(db_path):
    """
    Print the complete schema definition of an SQLite database.

    :param db_path: Path to the SQLite database file.
    """
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query to retrieve the schema for all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("No tables found in the database.")
            return
        
        print(f"Database Schema for {db_path}:")
        print("=" * 40)
        
        for table_name in tables:
            table_name = table_name[0]
            print(f"Table: {table_name}")
            print("-" * 40)
            
            # Query to retrieve the schema for the current table
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            if columns:
                print("Columns:")
                for col in columns:
                    print(f"  - {col[1]} ({col[2]}){' PRIMARY KEY' if col[5] else ''}")
            
            # Check for indices
            cursor.execute(f"PRAGMA index_list({table_name});")
            indices = cursor.fetchall()
            
            if indices:
                print("\nIndices:")
                for index in indices:
                    index_name = index[1]
                    unique = "UNIQUE" if index[2] else ""
                    print(f"  - {index_name} {unique}")
                    cursor.execute(f"PRAGMA index_info({index_name});")
                    index_columns = cursor.fetchall()
                    for idx_col in index_columns:
                        print(f"    - {idx_col[2]}")
            
            print("\n")
        
        print("=" * 40)
        print("Schema extraction complete.")
        
    except sqlite3.Error as e:
        print(f"Error: {e}")
    finally:
        if conn:
            conn.close()

# Example usage
db_path = "data/imdb.db"
print_schema(db_path)
