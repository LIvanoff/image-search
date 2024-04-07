from typing import Any

import pandas as pd
import numpy as np
import cv2
import copy
import os
from pathlib import Path

from src.config import cfg, cfg_from_yaml_file
from src.models.model import Model
from src.photos.models import Photo

config_folder = Path("../../config")
config_file = Path("config.yaml")
config_root = os.path.join(config_folder, config_file)
CFG_FILE_PATH = os.path.join((Path(__file__).resolve().parent / "./"), config_root)


class ModelLauncher:
    def __init__(self, task_type: str, lngs: list = None):
        self.config = self.parse_config(CFG_FILE_PATH)

        if task_type == "image_text_encoding":
            self.model_config = self.config.MODEL.IMAGE_TEXT_ENCODER
        elif task_type == "text_encoding":
            self.model_config = self.config.MODEL.TEXT_ENCODER
        elif task_type == "tagging":
            self.model_config = self.config.MODEL.DETECT
        elif task_type == "ocr":
            self.model_config = self.config.MODEL.OCR
        else:
            raise NotImplementedError(
                f"Задачи {task_type} не существует.\n"
                f"Вы можете указать одну из следующих задач: image_text_encoding, text_encoding or tagging"
            )

        self.model = Model(self.model_config, lngs=lngs)

    def text_to_image(self, output, input):
        # image = cv2.imread(file)
        image = np.array(input)
        for res in output:
            print(f'res: {res}')
            top_left = tuple(res[0][0])  # top left coordinates as tuple
            bottom_right = tuple(res[0][2])  # bottom right coordinates as tuple
            # draw rectangle on image
            image = cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), 2)
            # write recognized text on image (top_left) minus 10 pixel on y
            image = cv2.putText(image, res[1], (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.imshow(image)
        return image

    def parse_config(self, config_path):
        cfg_from_yaml_file(config_path, cfg)
        cfg.TAG = Path(config_path).stem
        cfg.EXP_GROUP_PATH = "/".join(
            config_path.split("/")[1:-1]
        )  # remove 'cfgs' and 'xxxx.yaml'
        return cfg

    def create_images_df(
            self, photos: list[Photo], filters: dict = None
    ) -> pd.DataFrame:
        columns = list()

        columns.append(self.model.output_name)
        columns.append("id")

        if filters is not None:
            columns += list(filters.keys())

        values = dict()
        values = {k: [] for k in columns}
        for photo in photos:
            photo = photo.__dict__
            for col in values.keys():
                if col in columns:
                    values[col].append(photo[col])

        images_db = pd.DataFrame.from_dict(values)
        return images_db

    # TODO:
    def find_images(self, input: str | Any, images_db: pd.DataFrame) -> pd.DataFrame:
        assert (
                self.model.task == "encoding"
        ), f"Функцию find_images() нельзя вызвать для модели {self.model.model_name}, она выполняет задачу {self.model.task}"

        output_vec = self.model(input)
        result_df = copy.deepcopy(images_db)
        result_df["distance"] = result_df.apply(
            lambda x: self.model.calculate_cos_dist(
                output_vec, x[f"{self.model.output_name}"]
            ),
            axis=1,
        )
        result_df_sorted = result_df.sort_values("distance").reset_index()
        result_df_sorted = result_df_sorted[["id", "distance"]]
        return result_df_sorted.head(self.model.topk)

    # TODO: на вход подается файл с изображением, на выходе list с тэгами
    def tagging(self, input):
        assert self.model.task == "tagging", (
            f"Функцию tagging() нельзя вызвать для модели {self.model.model_name}, "
            f"она выполняет задачу {self.model.task}"
        )
        output = self.model(input)
        return output

    # TODO: на вход подается файл с изображением или текст, или список, на выходе список (вектор) или список списков (список векторов)
    def vectorize(self, input):
        assert self.model.task == "encoding", (
            f"Функцию vectorize() нельзя вызвать для модели {self.model.model_name}"
            f", она выполняет задачу {self.model.task}"
        )
        output = self.model(input)
        return output

    def translate(self, input):
        output = self.model(
                            input,
                            paragraph=self.paragraph,
                            text_threshold=self.text_threshold
                            )
        return output

# model = ModelLauncher('image_text_encoding')
# text = 'Москва, 1980 г.'
# # text += 'car'
# out = model.vectorize(text)
# print(out.shape)
# print(type(out))
