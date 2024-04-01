from sentence_transformers import SentenceTransformer
from scipy import spatial
from PIL import Image
import pandas as pd
import numpy as np
import copy
import os
import argparse
from pathlib import Path

from config import cfg, cfg_from_yaml_file, log_config_to_file
from src.model import Model


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--img_path', type=str, help='image path')
    parser.add_argument('--query', type=str, help='text query')
    parser.add_argument('--cfg_file', type=str, required=True,
                        help='specify the config for inference modality type of choosen model')
    parser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--modality_type', type=str, required=True, help='multi or uni')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    return args, cfg


def main():
    global input
    args, cfg = parse_config()

    images_folder = cfg.IMAGES_FOLDER
    if args.modality_type == 'multi':
        cfg = cfg.MODEL.IMAGE_TEXT_ENCODER
        assert args.img_path is not None, 'Missing one of required argument: --img_path'
        input = args.img_path
    elif args.modality_type == 'uni':
        cfg = cfg.MODEL.TEXT_ENCODER
        assert args.query is not None, 'Missing one of required argument: --query'
        input = args.query

    model = Model(cfg)

    # if os.path.isfile(f'{name}.xlsx'):
    #     images_db = pd.read_excel(f'{name}.xlsx')
    # else:
    images_db = model.create_images_db(images_folder)

    print(images_db)

    result_df = model.found_similar_images(input, images_db)
    print(result_df)


if __name__ == '__main__':
    main()
