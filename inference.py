import pandas as pd
import copy
import os
from pathlib import Path

from config import cfg, cfg_from_yaml_file
from src.model import Model

config_folder = Path('config')
config_file = Path('config.yaml')
config_root = os.path.join(config_folder, config_file)
CFG_FILE_PATH = os.path.join((Path(__file__).resolve().parent / './'), config_root)


class ModelLauncher:
    def __init__(self, task_type: str):
        self.config = self.parse_config(CFG_FILE_PATH)
        self.img_folder = self.config.IMAGES_FOLDER

        if task_type == 'image_text_encoding':
            self.model_config = self.config.MODEL.IMAGE_TEXT_ENCODER
        elif task_type == 'text_encoding':
            self.model_config = self.config.MODEL.TEXT_ENCODER
        elif task_type == 'tagging':
            self.model_config = self.config.MODEL.DETECT
        else:
            raise NotImplementedError(f'Задачи {task_type} не существует.\n'
                                      f'Вы можете указать одну из следующих задач: image_text_enc, text_enc or tagging')

        self.model = Model(self.model_config)

    def parse_config(self, config_path):
        cfg_from_yaml_file(config_path, cfg)
        cfg.TAG = Path(config_path).stem
        cfg.EXP_GROUP_PATH = '/'.join(config_path.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
        return cfg

    def find_images(self, input: str, images_db: pd.DataFrame) -> pd.DataFrame:
        assert self.model.task == 'vectorize', f'Функцию find_images() нельзя вызвать для модели {self.model.model_name}, она выполняет задачу {self.model.task}'

        output_vec = self.model(input)
        result_df = copy.deepcopy(images_db)
        result_df['Distance_with_input'] = result_df.apply(
            lambda x: self.model.calculate_cos_dist(output_vec, x[f'{self.model.output_name}']), axis=1)
        result_df_sorted = result_df.sort_values('Distance_with_input').reset_index()
        result_df_sorted = result_df_sorted[['Image', 'Distance_with_input']]
        return result_df_sorted.head(self.model.topk)

    def tagging(self, input):
        assert self.model.task == 'tagging', f'Функцию tagging() нельзя вызвать для модели {self.model.model_name}, ' \
                                             f'она выполняет задачу {self.model.task}'
        output = self.model(input)
        return output

    def vectorize(self, input):
        assert self.model.task == 'encoding', f'Функцию vectorize() нельзя вызвать для модели {self.model.model_name}' \
                f', она выполняет задачу {self.model.task}'
        output = self.model(input)
        return output


# model = ModelLauncher('image_text_encoding')
# img = 'C:/Users/drfri/Pictures/0EmTWIPreJQ.jpg'
# out = model.vectorize(img)
# print(out.shape)
# print(type(out))