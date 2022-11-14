from PIL import Image
import requests
from io import BytesIO
import numpy as np
import pandas as pd
import os
import cv2
import gdown

df_path = 'data-active user-faceid-9112022.csv' 
df = pd.read_csv(df_path, encoding='ISO-8859-1')

route = 'dataset'
os.mkdir(route)

def url2img(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return None
        img = Image.open(BytesIO(response.content)) # already rgb
        img = np.array(img)
        return img
    except:
        return None

for index, row in df.iterrows():
    
    domain = row['domain']
    url_frontal_face = row['frontal_face']
    url_masked_face = row['masked_face']
    url_skewed_face = row['skewed_face']

    frontal_face = url2img(url_frontal_face)

    if frontal_face is None:
        print('frontal_face is None', domain)
        continue

    cv2.imwrite(f'{route}/{domain}_frontal_face.jpg', frontal_face[...,::-1])

url = "https://drive.google.com/file/d/1-aja0jCAdiFclVx_WDjJeUHdiHHBwM-J/view?usp=share_link"
output = "best_model_facepose_with_mask_EfficientNetV2S_192_512_2.h5"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)