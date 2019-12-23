#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 23:21:13 2019

@author: patrickmaus
"""

import os
import glob
from zipfile import ZipFile 

#%%
os.getcwd()

#%%
for file in glob.glob('/Users/patrickmaus/Documents/projects/AIS_data/2017/*AIS_2017_01_Zone10.zip'):
    print (file)
    with ZipFile(file, 'r') as zip: 
    # printing all the contents of the zip file 
        zip.printdir() 
      
        # extracting all the files 
        print('Extracting all the files now...') 
        zip.extractall('/Users/patrickmaus/Documents/projects/AIS_data/2017_unzipped') 
        print('Done!') 
        
#%%
 with ZipFile('my_python_files.zip','w') as zip: 
    # writing each file one by one 
    for file in file_paths: 
        zip.write(file) 