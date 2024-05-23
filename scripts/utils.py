import sys


def verify_destructive(message: str):    
    print(f"{message} [Y/n] ")
    response = input()
    if response and response.lower() != 'y':
        print("Aborting")
        sys.exit(1)
