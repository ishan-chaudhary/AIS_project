#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 23:21:13 2019

@author: patrickmaus
"""

import os
import glob
from zipfile import ZipFile 
import shutil

#%%
os.getcwd()

#%%
for file in glob.glob('/Users/patrickmaus/Documents/projects/AIS_data/2017/*.zip'):
    print (file)
    with ZipFile(file, 'r') as zip: 
    # printing all the contents of the zip file 
        zip.printdir() 
      
        # extracting all the files 
        print('Extracting files now...') 
        zip.extractall('/Users/patrickmaus/Documents/projects/AIS_data/2017_unzipped') 
        print('Done!') 
        

#%% shutil will delete the whole tree.  since this tree is created every time
## a file is unzipped, we can use it at the end of each file in the for loop 
## to free up memory
        
shutil.rmtree('/Users/patrickmaus/Documents/projects/AIS_data/2017_unzipped/AIS_ASCII_by_UTM_Month')