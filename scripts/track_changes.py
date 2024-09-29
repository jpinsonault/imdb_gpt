import os
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import List, Dict, Optional

import openai
from secret_config import openai_key

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training_script.log"),
        logging.StreamHandler()
    ]
)

# Initialize OpenAI Client
openai.api_key = openai_key
gpt_model = "gpt-4o-mini"

# Define Enums for States
class ScriptState(Enum):
    NO_CHANGES = "No changes detected."
    CHANGES_DETECTED = "Changes detected and committed."
    ERROR = "An error occurred."

# Background Prompt for GPT
background = """
I'm working on a silly experiment with neural networks, and I'm tweaking a lot of parameters, changing the structure, the layers, the activations, etc.
I want to be able to read the commit title and understand what I tweaked. I want a narrative of the changes, rather than a list. You'll be provided
with the git diff of the current changes, the current code, and the commit message history.
"""

def run_command(command: List[str], timeout: int = 30) -> Optional[str]:
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            timeout=timeout,
            text=True
        )
        logging.debug(f"Command '{' '.join(command)}' executed successfully.")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Command '{' '.join(command)}' failed with return code {e.returncode}")
        logging.error(f"stdout:\n{e.stdout}")
        logging.error(f"stderr:\n{e.stderr}")
    except subprocess.TimeoutExpired:
        logging.error(f"Command '{' '.join(command)}' timed out after {timeout} seconds.")
    return None

def get_git_diff(directory: str, timeout: int = 30) -> str:
    """Get git diff for a specific directory."""
    logging.info(f"Getting git diff for {directory}...")
    diff = run_command(['git', 'diff', directory], timeout)
    return diff if diff else ""

def get_recent_commit_history(limit: int = 5, timeout: int = 30) -> List[str]:
    """Retrieve recent git commit history."""
    logging.info("Retrieving recent commit history...")
    log = run_command(['git', 'log', f'--max-count={limit}'], timeout)
    return log.split('\n') if log else []

def get_changed_files(directory: str, timeout: int = 30) -> List[str]:
    """Get list of changed files in a directory."""
    logging.info(f"Getting changed files for {directory}...")
    files = run_command(['git', 'diff', '--name-only', directory], timeout)
    return [file for file in files.split('\n') if file] if files else []

def is_file_changed(file_path: str, timeout: int = 30) -> bool:
    """Check if a specific file has changes."""
    result = run_command(['git', 'diff', '--quiet', file_path], timeout)
    return result is None  # None means changes are present

def get_file_contents(file_path: str) -> str:
    """Read and return the contents of a file."""
    try:
        with open(file_path, 'r') as file:
            logging.debug(f"Reading contents of {file_path}.")
            return file.read()
    except IOError as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return ""

def generate_commit_message(diff: str, recent_commits: List[str], file_contents: Dict[str, str], folder_name: str) -> str:
    """Generate a commit message using OpenAI's GPT."""
    commit_history_str = "\n".join(recent_commits)
    file_contents_str = "\n\n".join([f"File: {path}\n\n{contents}" for path, contents in file_contents.items()])

    # Using your prompt exactly as provided
    prompt = f"""
    Background: {background}

    Describe the following changes in a commit message. Talk like a cracked 100x engineer at the top of your game hopped up on legal productivity drugs.

    message format: <timestamp from log folder name><short description as title><long description>

    Log folder name: {folder_name}

    File contents:
    {file_contents_str}

    Git diff: {diff}
    """

    try:
        response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": "You are an assistant that helps generate commit messages for code changes."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        commit_message = response.choices[0].message.content.strip()
        logging.debug(f"Generated commit message: {commit_message}")
        return commit_message
    except Exception as e:
        logging.error(f"Error generating commit message: {e}")
        return f"Update {folder_name}. Error autogenerating commit message."

def generate_folder_suggestion(diff: str, recent_commits: List[str]) -> str:
    """Generate a folder name suggestion using OpenAI's GPT."""

    # Using your folder name prompt exactly as provided
    prompt = f"""
    Background: {background}

    Suggest a simple folder name that reasonably captures the changes described below. be concise, we don't need prefixes, just
    something you'd expect an engineer to write for themselves like "added-another-dense-to-back_to_quarter" or
    "swapped-activation-to-relu-before-bottleneck".

    Git diff: {diff}
    """

    try:
        response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": "You are an assistant that helps generate folder names based on code changes."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        folder_name = response.choices[0].message.content.strip().replace('"', '').replace("'", "")
        # Sanitize and limit folder name length
        folder_name = sanitize_folder_name(folder_name)
        logging.debug(f"Suggested folder name: {folder_name}")
        return folder_name
    except Exception as e:
        logging.error(f"Error generating folder name: {e}")
        return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

def sanitize_folder_name(name: str, max_length: int = 50) -> str:
    """Sanitize the folder name by removing invalid characters and limiting its length."""
    # Replace spaces with underscores and remove invalid characters
    valid_name = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in name.replace(' ', '_'))
    # Truncate if necessary
    if len(valid_name) > max_length:
        valid_name = valid_name[:max_length]
    return valid_name

def add_all_changes(timeout: int = 30):
    """Stage all changes in git."""
    logging.info("Adding all changes to git...")
    run_command(['git', 'add', '--all'], timeout)

def commit_changes(commit_message: str, timeout: int = 30):
    """Commit staged changes with the provided commit message."""
    logging.info("Committing changes to git...")
    result = run_command(['git', 'commit', '-m', commit_message], timeout)
    if result:
        logging.info("Changes committed successfully.")
    else:
        logging.warning("No changes to commit.")

def push_changes(timeout: int = 30):
    """Push committed changes to the remote repository."""
    logging.info("Pushing changes to remote repository...")
    result = run_command(['git', 'push'], timeout)
    if result:
        logging.info("Changes pushed successfully.")
    else:
        logging.warning("Failed to push changes.")

def create_logs_dir(folder_name: str) -> Path:
    """Create a logs directory with the given folder name."""
    logs_dir = Path("logs") / f"{int(datetime.now().timestamp())}_{folder_name}"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Logs directory created at: {logs_dir}")
    except OSError as e:
        logging.error(f"Failed to create logs directory '{logs_dir}': {e}")
        raise
    return logs_dir

def suggest_folder_name() -> str:
    """Suggest a folder name based on current git diffs and commit history."""
    diff_scripts = get_git_diff('scripts/')
    diff_config = get_git_diff('config.py')
    combined_diff = diff_scripts + "\n" + diff_config
    recent_commits = get_recent_commit_history()

    if not combined_diff.strip():
        logging.info(ScriptState.NO_CHANGES.value)
        return "no_changes_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    folder_name = generate_folder_suggestion(combined_diff, recent_commits)
    return folder_name

def commit_changes_to_git(folder_name: str):
    """Commit changes to git with a generated commit message."""
    diff_scripts = get_git_diff('scripts/')
    diff_config = get_git_diff('config.py')
    combined_diff = diff_scripts + "\n" + diff_config

    if not combined_diff.strip():
        logging.info(ScriptState.NO_CHANGES.value)
        return ScriptState.NO_CHANGES.value

    changed_files = get_changed_files('scripts/')
    config_file = 'config.py'
    if is_file_changed(config_file):
        changed_files.append(config_file)

    file_contents = {file: get_file_contents(file) for file in changed_files}

    commit_message = generate_commit_message(combined_diff, get_recent_commit_history(), file_contents, folder_name)

    add_all_changes()
    commit_changes(commit_message)
    push_changes()

    logging.info(ScriptState.CHANGES_DETECTED.value)
    return ScriptState.CHANGES_DETECTED.value

def main():
    """Main function to orchestrate git change tracking and committing."""
    try:
        folder_name = suggest_folder_name()
        logs_dir = create_logs_dir(folder_name)
        state = commit_changes_to_git(folder_name)
        logging.info(f"Script finished with state: {state.value}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        logging.info(ScriptState.ERROR.value)

if __name__ == '__main__':
    main()
