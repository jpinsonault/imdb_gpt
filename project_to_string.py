import os
import sys

# Global variables
WHITELIST_EXTENSIONS = {'.py', '.js', '.html', '.css', '.md'}
BLACKLIST_FILES = {
    "attention_model.py", "clean_empty_logs.py", "convert_logs.py", "create_datasets.py", "create_normalized_tables.py",
    "create_search_tables.py", "create_vector_db.py", "do_stats_on_dataset.py", "download_imdb_dataset.py",
    "load_model.py", "load_tsvs.py", "look_at_log.py", "map_entity_ids_to_sql.py", "movie_stats.py", 
    "person_stats.py", "test.py", "track_changes.py", "train_imdb_llm.py", "train_imdb_siren.py", "print_db_schema.py",
    "utils.py", 'project_to_string.py'
}

BLACKLIST_FOLDERS = {'.git', 'pdfs', '__pycache__', 'venv'}

def prepare_project_string(directory):
    project_string = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        # Filter out blacklisted folders
        dirs[:] = [d for d in dirs if d not in BLACKLIST_FOLDERS]

        for file in files:
            # Get the file extension
            ext = os.path.splitext(file)[1]

            # Check if the file extension is whitelisted and the file is not blacklisted
            if ext in WHITELIST_EXTENSIONS and file not in BLACKLIST_FILES:
                # Construct the relative file path
                relative_path = os.path.relpath(os.path.join(root, file), directory)

                # Add file header and contents
                project_string.append(f"File: {relative_path}")
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        project_string.append(f.read())
                except Exception as e:
                    project_string.append(f"Error reading {relative_path}: {e}")

    # Join all collected strings into a single string
    return '\n\n'.join(project_string)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prepare_project_string.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]

    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)

    project_string = prepare_project_string(directory)

    print(project_string)
