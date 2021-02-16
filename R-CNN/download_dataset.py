import os
import requests
from tqdm import tqdm
import pandas as pd

def download_annotations(url, folder):
    buffer_size = 1024
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get("Content-Length", 0))
    filename = url.split("/")[-1]
    # progress bar, changing the unit to bytes instead of iteration (default by tqdm)
    progress = tqdm(response.iter_content(buffer_size), f"Downloading {filename}", total=file_size, unit="B", unit_scale=True, unit_divisor=1024)
    with open(f'{folder}/{filename}', "wb") as f:
        for data in progress:
            # write data read to the file
            f.write(data)
            # update the progress bar manually
            progress.update(len(data))

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def create_file_with_ids(filename, label, output):
    class_descriptions = pd.read_csv('data/oidv6-class-descriptions.csv')
    data = pd.read_csv(filename)
    data = data.join(class_descriptions.set_index('LabelName'), on='LabelName')

    pumpkin = data.loc[data['DisplayName']=='Banana', :]
    pumpkin_ids = pumpkin['ImageID'].tolist()

    with open(f'data/{output}.txt', 'w') as file:
        for id in pumpkin_ids:
            file.write(f'{output}/{id}\n')


def main():
    urls=[
      'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv',
      'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv',
      'https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv',
      'https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv']

    create_folder('data')

    for url in urls:
        download_annotations(url, 'data')

    create_file_with_ids('data/oidv6-train-annotations-bbox.csv', 'Banana', 'train')
    create_file_with_ids('data/validation-annotations-bbox.csv', 'Banana', 'validation')
    create_file_with_ids('data/test-annotations-bbox.csv', 'Banana', 'test')

main()