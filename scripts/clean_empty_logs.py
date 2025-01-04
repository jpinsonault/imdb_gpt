import sys
from pathlib import Path

def get_folder_size(folder):
    return sum(f.stat().st_size for f in folder.glob('**/*') if f.is_file())

def get_subfolder_sizes(directory):
    return {entry.name: get_folder_size(entry) for entry in directory.iterdir() if entry.is_dir()}

def truncate_string(string, max_length):
    if len(string) > max_length:
        return string[:max_length-3] + '...'
    return string

def print_formatted_table(subfolder_sizes):
    max_name_length = 30
    name_header = "Folder Name"
    size_header = "Size (MB)"
    
    print(f"{name_header:<{max_name_length}} | {size_header}")
    print("-" * (max_name_length + len(size_header) + 3))

    for subfolder, size in sorted(subfolder_sizes.items(), key=lambda item: item[1]):
        truncated_name = truncate_string(subfolder, max_name_length)
        size_mb = size / (1024 ** 2)
        print(f"{truncated_name:<{max_name_length}} | {size_mb:10.2f}")

def delete_smallest_folders(directory, subfolder_sizes, max_kilobytes, dry_run=False):
    total_deleted = 0
    folders_to_delete = []

    for subfolder, size in sorted(subfolder_sizes.items(), key=lambda item: item[1]):
        if size / 1024 <= max_kilobytes:
            folder_path = directory / subfolder
            folders_to_delete.append((subfolder, size, folder_path))
            total_deleted += size
        else:
            break

    if dry_run:
        print("Dry run: The following folders would be deleted:")
        for subfolder, size, _ in folders_to_delete:
            print(f"  {subfolder} ({size / (1024 ** 2):.2f} MB)")
        print(f"Total space that would be freed: {total_deleted / (1024 ** 2):.2f} MB")
    else:
        for subfolder, size, folder_path in folders_to_delete:
            try:
                for item in folder_path.glob('**/*'):
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        item.rmdir()
                folder_path.rmdir()
                print(f"Deleted folder: {subfolder}")
                del subfolder_sizes[subfolder]
            except Exception as e:
                print(f"Error deleting {subfolder}: {e}")
        
        print(f"Total space freed: {total_deleted / (1024 ** 2):.2f} MB")

    return total_deleted

def main():
    if len(sys.argv) < 2:
        print("Please provide a directory path.")
        return

    directory = Path(sys.argv[1])
    if not directory.is_dir():
        print(f"{directory} is not a valid directory.")
        return
    
    subfolder_sizes = get_subfolder_sizes(directory)

    print("Current folder sizes:")
    print_formatted_table(subfolder_sizes)

    # Dry run
    print("\nPerforming dry run:")
    delete_smallest_folders(directory, subfolder_sizes.copy(), 100, dry_run=True)

    # Ask user if they want to proceed with actual deletion
    user_input = input("\nDo you want to proceed with deletion? (yes/no): ").lower()
    if user_input == 'yes':
        delete_smallest_folders(directory, subfolder_sizes, 100, dry_run=False)
        print("\nAfter deletion:")
        print_formatted_table(subfolder_sizes)
    else:
        print("Deletion cancelled.")

if __name__ == "__main__":
    main()