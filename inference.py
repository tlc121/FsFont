"""
#   Copyright (C) 2022 BAIDU CORPORATION. All rights reserved.
#   Author     :   tanglicheng@baidu.com
#   Date       :   2022-02-10
"""
import argparse
import os
from tqdm import tqdm
import json
import lmdb
import random
from PIL import Image, ImageFont, ImageDraw
import io
import paddle
from glob import glob
from evaluator import *
from models import generator_dispatch
from sconf import Config
import shutil
from transform import setup_transforms
from datasets import load_json, get_fixedref_loader
from build_dataset.build_dataset import save_lmdb


def getCharList(root):
    '''
    getCharList
    '''
    charlist = []
    for img_path in glob(root + '/*.jpg') + glob(root + '/*.png'):
        ch = os.path.basename(img_path).split('.')[0]
        charlist.append(ch)
    return charlist
        

def getMetaDict(image_root_list, content_name, cr_mapping):
    meta_dict = dict()
    random_sample_list = []
    for idx, file_path in enumerate(image_root_list):
        font_name = os.path.basename(file_path)
        if font_name == content_name:
            content_path = file_path
            continue
        all_ch_list = getCharList(file_path)
        meta_dict[font_name] = {
            "path": file_path,
            "charlist": None
        }
        meta_dict[font_name]["charlist"] = all_ch_list
        
        #adaptively choose the possible inference unicodes according to the style unicodes you have.
        infer_unis = []
        style_set = set(hex(ord(ch))[2:].upper() for ch in all_ch_list)
        for uni in cr_mapping.keys():
            uni_pair = set(cr_mapping[uni])
            if len(set.intersection(uni_pair, style_set)) < len(uni_pair):
                continue
            infer_unis.append(chr(int(uni, 16)))
    
    meta_dict[content_name] = {
            "path": content_path, 
            "charlist": None
        }
    meta_dict[content_name]["charlist"] = infer_unis
    return meta_dict


def build_meta4build_dataset(meta_path, img_path_list, content_name, cr_mapping):
    '''
    build_meta4build_dataset
    '''
    out_dict_path = meta_path
    out_dict = getMetaDict(img_path_list, content_name, cr_mapping)
    with open(out_dict_path, 'w') as fout:
        json.dump(out_dict, fout, indent=4, ensure_ascii=False)
    print("dataset meta:", out_dict_path)


def build_dataset4inference(target_img_path, meta_path, content_root, lmdb_path, json_path, cr_mapping):
    '''
    target_img_path: test image directory
    content_root: content image directory
    meta_path: {font1: {root: '/path/', 'charlist': [ch1, ch2, ch3..]}, font2: ....}
    lmdb_path: lmdb directory
    json_path: {font1: [uni1, uni2, uni3, ..], font2: [uni1, uni2, uni3, ...]}
    '''
    ### START
    img_path_list = [target_img_path] + [content_root] 
    content_name = os.path.basename(content_root)
    build_meta4build_dataset(meta_path, img_path_list, content_name, cr_mapping)
    with open(meta_path) as f:
        fpc_meta = json.load(f)
    valid_dict = save_lmdb(lmdb_path, fpc_meta)
    with open(json_path, "w") as f:
        json.dump(valid_dict, f)
    print("lmdb_path:", lmdb_path)
    print("test meta:", json_path)
    ### END
    

def build_testmeta4inference(target_name, target_root, content_name="kaiti_xiantu"):
    '''
    build_testmeta4inference
    '''
    meta_file = os.path.join(target_root, "dataset_meta.json")
    save_path = os.path.join(target_root, "test.json")
    avali_set = {}
    
    with open(meta_file, 'r') as fin:
        original_meta = json.load(fin)
    
    target_ori_unis = original_meta[target_name]
    
    # build test meta file
    test_dict = {
        "gen_fonts": [target_name],
        "gen_unis": original_meta[content_name],
        "ref_unis": target_ori_unis
    }
    with open(save_path, 'w') as fout:
        json.dump(test_dict, fout, ensure_ascii=False, indent=4)
    print("test metafile save to ", save_path)
    return save_path, avali_set


def eval_ckpt(args, cfg, avail, target_root):
    '''
    eval_ckpt
    '''
    logger = Logger.get()

    content_name = cfg.content_name
    trn_transform, val_transform = setup_transforms(cfg)

    env = load_lmdb(cfg.data_path)
    env_get = lambda env, x, y, transform: transform(read_data_from_lmdb(env, f'{x}_{y}')['img'])
    test_meta = load_json(args.test_meta)
    
    g_kwargs = cfg.get('g_args', {})
    g_cls = generator_dispatch()
    gen = g_cls(1, cfg['C'], 1, cfg, **g_kwargs)
    if cfg.use_half:
        gen.half()
    gen.to('gpu')
    
    weight = paddle.load(args.weight)
        
    if "generator_ema" in weight:
        weight = weight["generator_ema"]
    gen.set_state_dict(weight)
    logger.info(f"load checkpoint from {args.weight}")
    writer = None

    evaluator = Evaluator(
                          env,
                          env_get,
                          cfg, 
                          logger,
                          writer,
                          cfg["batch_size"],
                          val_transform,
                          content_name,
                          use_half=cfg.use_half
                          )

    img_dir = Path(target_root)
    ref_unis = test_meta["ref_unis"]
    gen_unis = test_meta["gen_unis"]
    gen_fonts = test_meta["gen_fonts"]
    target_dict = {f: gen_unis for f in gen_fonts}
    loader = get_fixedref_loader(env=env,
                                 env_get=env_get,
                                 target_dict=target_dict,
                                 ref_unis=ref_unis,
                                 cfg=cfg,
                                 transform=val_transform,
                                 num_workers=cfg.n_workers,
                                 shuffle=False
                                 )[1]


    logger.info("Save CV results to {} ...".format(img_dir))
    saving_folder = evaluator.save_each_imgs(gen, loader, args.img_path, save_dir=target_root, reduction='mean')
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_paths", nargs="+", help="path to config.yaml")
    parser.add_argument("--weight", help="path to weight to evaluate")
    parser.add_argument("--content_font", help="path to content font")
    parser.add_argument("--img_path", help="path of the your test img directory.")
    parser.add_argument("--saving_root", help="saving directory.")
    
    args = parser.parse_args()
    cfg = Config(*args.config_paths, default="./cfgs/defaults.yaml")
    
    
    target_folder = args.img_path 
    content_root = args.content_font
    saving_root = args.saving_root
    content_name = os.path.basename(content_root)
    target_name = os.path.basename(target_folder)
    target_root = os.path.join(saving_root, target_name)
    
    with open(cfg.content_reference_json, 'r') as f:
        cr_mapping = json.load(f)
    
    #create directory
    os.makedirs(target_root, exist_ok=True)

    # lmdb directory
    lmdb_path = os.path.join(target_root, "lmdb")
    if os.path.exists(lmdb_path):
        shutil.rmtree(lmdb_path)
    os.makedirs(lmdb_path, exist_ok=True)
    

    # meta file
    json_path = os.path.join(target_root, "dataset_meta.json")
    meta_path = os.path.join(target_root, "ori_dataset_meta.json")

    build_dataset4inference(target_folder, meta_path, content_root, lmdb_path, json_path, cr_mapping)

    ###test.json
    save_path, avail = build_testmeta4inference(target_name, target_root, content_name)
    args.test_meta = save_path

    cfg.content_name = content_name
    cfg.data_path = lmdb_path

    print("======> Test Params:")
    print("test meta:", args.test_meta)
    print("content font:", cfg.content_name)
    print("lmdb datasets:", cfg.data_path)

    eval_ckpt(args, cfg, avail, target_root)
