import os
from typing import Tuple


def count_files_recursive(path) -> Tuple[int, int]:
    file_count = 0
    directory_count = 0
    for root, dirs, files in os.walk(path):
        file_count += len(files)
        directory_count += len(dirs)
    return file_count, directory_count


