import pandas as pd
import requests
import zipfile
import os

#%%
def download_url(link, download_path, unzip_path, chunk_size=128):
    print('Testing link...')
    r = requests.get(link, stream=True)
    if r.status_code == 200:
        print('Link good for {}!'.format(file_name))
    else:
        print('Link did not return 200 status code')
    with open(download_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
    print('File downloaded.')
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(path=unzip_path)
    print('File unzipped!')
    # delete the zipped file
    os.remove(download_path)
    print('Zip file deleted.')
    print()

#%% set variables
year = 2017
month = 1
zone = 1

folder = '/Users/patrickmaus/Documents/projects/AIS_data'
file_name = f'{str(year)}_{str(month).zfill(2)}_{str(zone).zfill(2)}'
download_path = f'{folder}/{file_name}.zip'
unzip_path = f'{folder}/{str(year)}'
link = ('https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{0}/AIS_{0}_{1}_Zone{2}.zip'
        .format(str(year), str(month).zfill(2), str(zone).zfill(2)))

download_url(link, download_path, unzip_path)

#%%
folder = '/Users/patrickmaus/Documents/projects/AIS_data'
for z in [9, 10, 11, 14, 15, 16, 17, 18, 19, 20]:
    for m in range(2, 3):
        file_name = f'{str(year)}_{str(month).zfill(2)}_{str(zone).zfill(2)}'
        download_path = f'{folder}/{file_name}.zip'
        unzip_path = f'{folder}/{str(year)}'
        link = ('https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{0}/AIS_{0}_{1}_Zone{2}.zip'
                .format(str(year), str(month).zfill(2), str(zone).zfill(2)))


