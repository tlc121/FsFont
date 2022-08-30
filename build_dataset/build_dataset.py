"""
#   Copyright (C) 2022 BAIDU CORPORATION. All rights reserved.
#   Author     :   tanglicheng@baidu.com
#   Date       :   2022-02-10
"""
import argparse
import json
import io
import os
import lmdb
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm
import numpy as np
import cv2
import shutil

def save_lmdb(env_path, font_path_char_dict):
    """[saving lmdb]

    Args:
        env_path (string): folder root
        font_path_char_dict (list): img lists in folder

    Returns:
        [json]: {font name: [ch1, ch2, ch3, ch4, ....]}
    """
    env = lmdb.open(env_path, map_size=1024 ** 4)
    valid_dict = {}
    for fname in tqdm(font_path_char_dict):
        fontpath = font_path_char_dict[fname]["path"]
        charlist = font_path_char_dict[fname]["charlist"]
        unilist = []
        for char in charlist:
            img_path = os.path.join(fontpath, char+'.png')
            if not os.path.exists(img_path):
                img_path = os.path.join(fontpath, char+'.jpg')
            
            uni = hex(ord(char))[2:].upper()
            unilist.append(uni)
            char_img = cv2.imread(img_path, 0)
            #char_img = cv2.resize(char_img, (128, 128))

            char_img = Image.fromarray(char_img)
            img = io.BytesIO()
            char_img.save(img, format="PNG")
            img = img.getvalue()
            lmdb_key = f"{fname}_{uni}".encode("utf-8")

            with env.begin(write=True) as txn:
                txn.put(lmdb_key, img)

        valid_dict[fname] = unilist

    return valid_dict

