from datetime import datetime
import os
import subprocess
from openai import OpenAI
from secret_config import openai_key

client = OpenAI(api_key=openai_key)
gpt_model = "gpt-4o-mini"

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
        print(f"Command '{' '.join(command)}' failed with return code {e.returncode}")
        print(f"stdout:\n{e.stdout}")
        print(f"stderr:\n{e.stderr}")
        return None
    except subprocess.TimeoutExpired:
        print(f"Command '{' '.join(command)}' timed out")
        return None

def get_git_diff(directory, timeout=30):
    print(f"Getting git diff for {directory}...")
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
        print(f"Error reading file {file_path}: {e}")
        return ""

def generate_commit_message(diff, recent_commits, file_contents, folder_name) -> str:
    commit_history_str = "\n".join(recent_commits)
    file_contents_str = "\n\n".join([f"File: {path}\n\n{contents}" for path, contents in file_contents.items()])

    prompt = f"""
    Background: {background}

    Describe the following changes in a commit message. Don't assume my intent, just describe the changes as if to a blind person

    message format: <title><newline><newline><description><haiku about the changes><log folder name>

    Log folder name: {folder_name}

    File contents:
    {file_contents_str}

    Recent commit history:
    {commit_history_str}

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
    commit_history_str = "\n".join(recent_commits)

    prompt = f"""
    Background: {background}

    Suggest a simple folder name that reasonably captures the changes described below. be concise, we don't need prefixes, just
    something you'd expect an engineer to write for themselves like "added-another-dense-to-back_to_quarter" or
    "swapped-activation-to-relu-before-bottleneck".

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

def add_all_changes(timeout=30):
    print("Adding all changes...")
    run_command(['git', 'add', '--all'], timeout)

def commit_changes(commit_message, timeout=30):
    print(f"Committing changes\n-----{commit_message}\n-----")
    run_command(['git', 'commit', '-m', commit_message], timeout)

def push_changes(timeout=30):
    print("Pushing changes to remote...")
    run_command(['git', 'push'], timeout)

def suggest_folder_name() -> str:
    diff = get_git_diff('scripts/')
    diff += get_git_diff('config.py')
    recent_commits = get_recent_commit_history()

    try:
        return generate_folder_suggestion(diff, recent_commits)
    except Exception as e:
        print(f"Error generating folder name: {e}")
        return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

def commit_changes_to_git(folder_name: str):
    print("Committing changes to git...")
    diff = get_git_diff('scripts/')
    recent_commits = get_recent_commit_history()

    if not diff:
        print("No changes detected in the scripts directory.")
        return

    config_file = './config.py'
    changed_files = get_changed_files('scripts/')

    if is_file_changed('./config.py'):
        changed_files.append(config_file)

    file_contents = {file: get_file_contents(file) for file in changed_files}

    try:
        commit_message = generate_commit_message(diff, recent_commits, file_contents, folder_name)
    except Exception as e:
        print(f"Error generating commit message: {e}")
        commit_message = f"Update {folder_name}. Error autogenerating commit message."

    add_all_changes()
    commit_changes(commit_message)

    print(f"Committed")

if __name__ == '__main__':
    folder_name = suggest_folder_name()
    commit_changes_to_git(folder_name)
