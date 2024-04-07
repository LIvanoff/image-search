from typing import Any

import pandas as pd
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
    def __init__(self, task_type: str):
        self.config = self.parse_config(CFG_FILE_PATH)

        if task_type == "image_text_encoding":
            self.model_config = self.config.MODEL.IMAGE_TEXT_ENCODER
        elif task_type == "text_encoding":
            self.model_config = self.config.MODEL.TEXT_ENCODER
        elif task_type == "tagging":
            self.model_config = self.config.MODEL.DETECT
        else:
            raise NotImplementedError(
                f"Задачи {task_type} не существует.\n"
                f"Вы можете указать одну из следующих задач: image_text_encoding, text_encoding or tagging"
            )

        self.model = Model(self.model_config)

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


# model = ModelLauncher('image_text_encoding')
# text = 'Москва, 1980 г.'
# # text += 'car'
# out = model.vectorize(text)
# print(out.shape)
# print(type(out))
