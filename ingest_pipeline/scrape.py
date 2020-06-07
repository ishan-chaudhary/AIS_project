import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import date
import requests
import zipfile



#%% set variables
year = 2017
month = 1
zone = 1
file_name = '{}_{}_{}'.format(str(year), str(month).zfill(2), str(zone).zfill(2))

download_path = '/Users/patrickmaus/Documents/projects/AIS_data/{}/{}.zip'.format(str(year), file_name)
unzip_path = '/Users/patrickmaus/Documents/projects/AIS_data/{}/'.format(str(year))


link = ('https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{0}/AIS_{0}_{1}_Zone{2}.zip'
        .format(str(year), str(month).zfill(2), str(zone).zfill(2)))

#%%
def download_url(link, download_path, unzip_path, chunk_size=128):
    print('Testing link...')
    response = requests.get(link)
    if response.status_code == 200:
        print('Link good!')
    else:
        print('Link did not return 200 status code')
    r = requests.get(link, stream=True)
    with open(download_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
    print('File downloaded.')
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(path=unzip_path)
    print('File unzipped!')

download_url(link, download_path, unzip_path)



