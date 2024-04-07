import os

import numpy as np
import easyocr
import pandas as pd
from scipy import spatial

from ultralytics import YOLO
from sentence_transformers import SentenceTransformer


class Model:
    def __init__(self, config: str, lngs: list) -> None:
        self.config = config
        self.model_name = self.config.NAME
        self.db_name = self.config.OUTPUT
        self.output_name = self.config.OUTPUT
        self.task = self.config.TASK

        if self.task == 'tagging':
            self.tags_dict = self.inverse_tags(self.config.TAGS)
            self.classes = list(self.tags_dict.keys())
            self.conf = self.config.CONF
            yolo_wts = self.model_name + '.pt'
            self.model = YOLO(yolo_wts)
        elif self.task == 'encoding':
            self.topk = self.config.TOPK
            self.model = SentenceTransformer(self.model_name)
        else:
            self.device = self.config.DEVICE
            is_gpu = True if self.device == 'gpu' else False
            self.model = easyocr.Reader(lngs, gpu=is_gpu)

        self.forward = self.setup_forward()

    def __vectorize_img(self, img) -> np.array:
        '''
        :param img: file
        :return: list
        '''
        return self.model.encode(img)

    def __vectorize_text(self, text: str) -> np.array:
        return self.model.encode(text)

    def inverse_tags(self, tags_dict):
        new_tags_dict = {v: k for k, v in tags_dict.items()}
        return new_tags_dict

    def setup_forward(self):
        forward = {}
        if self.output_name == 'image_vector':
            forward[self.output_name] = self.__vectorize_img
        elif self.output_name == 'text_vector':
            forward[self.output_name] = self.__vectorize_text
        elif self.output_name == 'text':
            forward[self.output_name] = self.model.readtext
        else:
            forward[self.output_name] = self.__detect
        return forward

    def __detect(self, img):
        '''
        функция инференса YOLO
        :return: list
        '''
        result = self.model.predict(img, classes=self.classes, conf=self.conf)
        cls_id = [int(x) for x in result[0].boxes.cls.cpu()]
        cls = self.id_to_names(cls_id)
        return cls

    def id_to_names(self, cls_id):
        names = []
        for id in cls_id:
            if id in self.tags_dict.keys() and self.tags_dict[id] not in names:
                names.append(self.tags_dict[id])
        return names

    def get_df(self, df_path: str) -> pd.DataFrame:
        data_df = pd.read_json(df_path)
        data_df[f'{self.output_name}'] = data_df[f'{self.output_name}'].apply(lambda x: np.array(x))
        return data_df

    @staticmethod
    def calculate_cos_dist(emb_a: np.array, emb_b: np.array) -> float:
        result_distance = spatial.distance.cosine(emb_a, emb_b)
        return result_distance

    def __call__(self, input, *args, **kwargs):
        out = self.forward[self.output_name](input)
        return out
