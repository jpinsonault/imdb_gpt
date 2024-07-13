import sys


def verify_destructive(message: str):    
    print(f"{message} [Y/n] ")
    response = input()
    if response and response.lower() != 'y':
        print("Aborting")
        sys.exit(1)


def print_project_config(config):
    print("Config:")
    for key, value in config.items():
        print(f"  {key}: {value}")