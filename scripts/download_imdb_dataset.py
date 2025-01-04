# download_imdb_dataset.py

import gzip
import os
from pathlib import Path
import shutil

import requests
from config import project_config
from tqdm import tqdm


def extract_and_rename_gzip(input_gzip_path, output_dir=None):
    # Determine the base filename (without .gz extension)
    base_filename = os.path.basename(input_gzip_path).replace('.gz', '')

    # Set the output directory to the current directory if not specified
    if output_dir is None:
        output_dir = os.path.dirname(input_gzip_path)

    # Construct the path for the extracted file
    extracted_file_path = os.path.join(output_dir, 'data.tsv')

    # Construct the path for the renamed file
    renamed_file_path = os.path.join(output_dir, base_filename)

    print(f"Extracting {input_gzip_path}...")
    with gzip.open(input_gzip_path, 'rb') as f_in:
        with open(extracted_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Rename the extracted file
    os.rename(extracted_file_path, renamed_file_path)

    return renamed_file_path

def download_imdb_datasets(data_dir: Path, base_url, file_names):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for file_name in file_names:
        url = base_url + file_name
        tqdm.write(f"Downloading {file_name}")

        response = requests.get(url, stream=True)
        file_path = os.path.join(data_dir, file_name)

        if response.status_code == 200:
            with open(file_path, 'wb') as f, tqdm(
                unit='B', unit_scale=True, unit_divisor=1024,
                total=int(response.headers.get('content-length', 0)),
                desc=file_name
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
                    bar.update(len(chunk))
        else:
            print(f"Failed to download {file_name}")

            
            
if __name__ == '__main__':
    data_dir = Path(project_config['data_dir']) / 'imdb_tsvs'
    base_url = 'https://datasets.imdbws.com/'
    file_names = [
        'name.basics.tsv.gz',
        'title.akas.tsv.gz',
        'title.basics.tsv.gz',
        'title.crew.tsv.gz',
        'title.episode.tsv.gz',
        'title.principals.tsv.gz',
        'title.ratings.tsv.gz'
    ]
    
    download_imdb_datasets(data_dir, base_url, file_names)

    for file_name in file_names:
        file_path = data_dir / file_name
        extract_and_rename_gzip(file_path, data_dir)
        os.remove(file_path)