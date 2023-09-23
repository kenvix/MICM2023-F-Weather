import os


def count_files_recursive(path, file_only=True):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file_only and not os.path.isdir(os.path.join(root, file)):
                count += 1
    return count
