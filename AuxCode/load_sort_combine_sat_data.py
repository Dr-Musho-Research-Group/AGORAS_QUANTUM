# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 16:50:16 2021

@author: tempker
"""
from pathlib import Path
import os
import re


path = Path(r'C:\Users\tempker\Documents\School')
os.chdir(path)

path = Path.cwd().joinpath('composite_images')
images = Path(path).glob("*.png")
image_strings = [str(p) for p in images]


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

image_strings.sort(key=natural_keys)
#images = [img_to_array(load_img(x)) for x in image_strings]

import pandas as pd

data = pd.read_csv('model_data.csv')
data['path'] = pd.Series(image_strings)

data.to_csv('model_data.csv')