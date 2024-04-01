from sentence_transformers import SentenceTransformer
from scipy import spatial
from PIL import Image
import pandas as pd
import numpy as np
import copy
import os
from pathlib import Path

from config import cfg, cfg_from_yaml_file, log_config_to_file
from src.model import Model

config_folder = Path('config')
config_file = Path('config.yaml')
config_root = os.path.join(config_folder, config_file)
CFG_FILE_PATH = os.path.join((Path(__file__).resolve().parent / './'), config_root)


class ModelLauncher:
    def __init__(self, task_type: str):
        self.config = self.parse_config(CFG_FILE_PATH)
        self.img_folder = self.config.IMAGES_FOLDER

        if task_type == 'image_text_emb':
            self.model_config = self.config.MODEL.IMAGE_TEXT_ENCODER
        elif task_type == 'text_emb':
            self.model_config = self.config.MODEL.TEXT_ENCODER
        # elif task_type == 'auto_markup':
        #     self.model_config = None
        else:
            raise NotImplementedError(f'Task {task_type} not implemented.\n'
                                      f'You can specify the following tasks: image_text_emb, text_emb or auto_markup')

        self.model = Model(self.config)

    def __call__(self, input, *args, **kwargs):
        output = self.model.vectorize_img()
        return output

    def parse_config(self, config_path):
        cfg_from_yaml_file(config_path, cfg)
        cfg.TAG = Path(config_path).stem
        cfg.EXP_GROUP_PATH = '/'.join(config_path.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
        return cfg

    def inference(self, x):
        embedding = self.model(x)
