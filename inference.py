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

    def find_images(self, input: str, images_db: pd.DataFrame) -> pd.DataFrame:
        input_vec = self.model(input)
        result_df = copy.deepcopy(images_db)
        result_df['Distance_with_input'] = result_df.apply(
            lambda x: self.model.calculate_cos_dist(input_vec, x[f'{self.model.embedding_name}']), axis=1)
        result_df_sorted = result_df.sort_values('Distance_with_input').reset_index()
        result_df_sorted = result_df_sorted[['Image', 'Distance_with_input']]
        return result_df_sorted.head(self.model.topk)
