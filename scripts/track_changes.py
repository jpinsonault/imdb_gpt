import logging
from datetime import datetime
import os
import subprocess
from openai import OpenAI
from secret_config import openai_key
from pathlib import Path
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

client = OpenAI(api_key=openai_key)
gpt_model = "gpt-4o-mini"

class State(Enum):
    NO_CHANGES = "No changes detected"
    CHANGES_DETECTED = "Changes detected"
    ERROR = "Error occurred"
    SUCCESS = "Success"

background = """
I'm working on a silly experiment with neural networks, and I'm tweaking a lot of parameters, changing the structure, the layers, the activations, etc.
I want to be able to read the commit title and understand what I tweaked. I want a narrative of the changes, rather than a list. You'll be provided
with the git diff of the current changes, the current code, and the commit message history.
"""

def run_command(command, timeout=30):
    try:
        result = subprocess.run(command, check=True, capture_output=True, timeout=timeout, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command '{' '.join(command)}' failed with return code {e.returncode}")
        logger.error(f"stdout:\n{e.stdout}")
        logger.error(f"stderr:\n{e.stderr}")
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"Command '{' '.join(command)}' timed out")
        return None

def get_git_diff(directory, timeout=30):
    logger.info(f"Getting git diff for {directory}...")
    return run_command(['git', 'diff', directory], timeout)

def get_recent_commit_history(limit=5, timeout=30):
    return run_command(['git', 'log', f'--max-count={limit}'], timeout).split('\n')

def get_changed_files(directory, timeout=30):
    files = run_command(['git', 'diff', '--name-only', directory], timeout)
    return [file for file in files.split('\n') if file] if files else []

def is_file_changed(file_path, timeout=30):
    return run_command(['git', 'diff', '--quiet', file_path], timeout) is None

def get_file_contents(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except IOError as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""

def generate_commit_message(diff, recent_commits, file_contents, folder_name) -> str:
    commit_history_str = "\n".join(recent_commits)
    file_contents_str = "\n\n".join([f"File: {path}\n\n{contents}" for path, contents in file_contents.items()])

    prompt = f"""
    Background: {background}

    Describe the following changes in a commit message. Talk like a cracked 100x engineer at the top of your game hopped up on legal productivity drugs.

    message format: <timestamp from log folder name><short description as title><long description>

    Log folder name: {folder_name}

    File contents:
    {file_contents_str}

    Git diff: {diff}
    """

    response = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": "You are an assistant that helps generate commit messages for code changes."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

def generate_folder_suggestion(diff, recent_commits):
    prompt = f"""
    Background: {background}

    Suggest a simple folder name that reasonably captures the changes described below. Be concise, something you'd expect an engineer to write
    for themselves like "added-another-dense-to-back_to_quarter" or "swapped-activation-to-relu-before-bottleneck".

    Git diff: {diff}
    """

    response = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": "You are an assistant that helps generate folder names based on code changes."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    return response.choices[0].message.content.strip().replace('"', '')

def create_logs_dir(folder_name):
    # Truncate folder name if it's too long
    truncated_folder_name = folder_name[:50]
    logs_dir = Path(f'logs/{truncated_folder_name}')
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Logs directory created: {logs_dir}")
        return logs_dir
    except OSError as e:
        logger.error(f"Error creating logs directory: {e}")
        raise

def add_all_changes(timeout=30):
    logger.info("Adding all changes...")
    run_command(['git', 'add', '--all'], timeout)

def commit_changes(commit_message, timeout=30):
    logger.info(f"Committing changes\n-----{commit_message}\n-----")
    run_command(['git', 'commit', '-m', commit_message], timeout)

def push_changes(timeout=30):
    logger.info("Pushing changes to remote...")
    run_command(['git', 'push'], timeout)

def suggest_folder_name() -> str:
    diff = get_git_diff('scripts/')
    diff += get_git_diff('config.py')
    recent_commits = get_recent_commit_history()

    try:
        return generate_folder_suggestion(diff, recent_commits)
    except Exception as e:
        logger.error(f"Error generating folder name: {e}")
        return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

def commit_changes_to_git(folder_name: str):
    logger.info("Committing changes to git...")
    diff = get_git_diff('scripts/')
    recent_commits = get_recent_commit_history()

    if not diff:
        logger.info(State.NO_CHANGES.value)
        return

    config_file = './config.py'
    changed_files = get_changed_files('scripts/')

    if is_file_changed(config_file):
        changed_files.append(config_file)

    file_contents = {file: get_file_contents(file) for file in changed_files}

    try:
        commit_message = generate_commit_message(diff, recent_commits, file_contents, folder_name)
        logger.info(State.SUCCESS.value)
    except Exception as e:
        logger.error(f"Error generating commit message: {e}")
        commit_message = f"Update {folder_name}. Error autogenerating commit message."

    add_all_changes()
    commit_changes(commit_message)

if __name__ == '__main__':
    folder_name = suggest_folder_name()
    try:
        create_logs_dir(folder_name)
        commit_changes_to_git(folder_name)
        push_changes()
    except Exception as e:
        logger.error(f"An error occurred during the process: {e}")
