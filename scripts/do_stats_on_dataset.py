from collections import defaultdict
import json
from config import project_config


from pathlib import Path


def do_stats_on_dataset(files_by_types):
    """give me the percentiles of the length of the 'result' field in the jsonl files
    """

    length_counts = defaultdict(int)
    for file in files_by_types:
        file_path = data_dir / file
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_dict = json.loads(line)
                result_len = len(line_dict['title'])
                length_counts[result_len] += 1

    print("length_counts", length_counts)


if __name__ == '__main__':
    data_dir = Path(project_config['data_dir'])

    files_by_types = [
        "movie.jsonl",
    ]

    do_stats_on_dataset(files_by_types)

