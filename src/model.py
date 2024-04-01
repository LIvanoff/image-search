from sentence_transformers import SentenceTransformer
from scipy import spatial
from PIL import Image
import pandas as pd
import numpy as np
import copy
import os


class Model:
    def __init__(self, config: str, topk: int = 20) -> None:
        self.config = config
        self.topk = topk
        self.model_name = self.config.NAME
        self.model = SentenceTransformer(self.model_name)
        self.db_name = self.config.EMBEDDING_NAME
        self.embedding_name = self.config.EMBEDDING_NAME

    def vectorize_img(self, img_path: str) -> np.array:
        img = Image.open(img_path)
        return self.model.encode(img)

    def vectorize_text(self, text: str) -> np.array:
        return self.model.encode(text)

    def auto_markup(self, img):
        '''
        функция инференса YOLO
        :return:
        '''
        pass

    def create_images_db(self, images_folder: str) -> pd.DataFrame:
        data_dict = dict()
        for file_name in os.listdir(images_folder):
            image_path = os.path.join(images_folder, file_name)
            if os.path.isfile(image_path):
                emb = self.vectorize_img(image_path)
                data_dict[file_name] = emb
        images_db = pd.DataFrame(data_dict.items(), columns=['Image', self.embedding_name])
        images_db.to_excel(f"{self.db_name}.xlsx")
        return images_db

    def get_df(self, df_path: str) -> pd.DataFrame:
        data_df = pd.read_json(df_path)
        data_df[f'{self.embedding_name}'] = data_df[f'{self.embedding_name}'].apply(lambda x: np.array(x))
        return data_df

    @staticmethod
    def calculate_cos_dist(emb_a: np.array, emb_b: np.array) -> float:
        result_distance = spatial.distance.cosine(emb_a, emb_b)
        return result_distance

    '''надо переписать под разные модели'''
    def found_similar_images(self, input_img_path: str, images_db: pd.DataFrame) -> pd.DataFrame:
        input_vec = self.vectorize_img(input_img_path)
        result_df = copy.deepcopy(images_db)
        import time
        time_start = time.time()
        result_df['Distance_with_input'] = result_df.apply(lambda x: self.calculate_cos_dist(input_vec, x[f'{self.embedding_name}']), axis=1)
        print(time.time() - time_start)
        result_df_sorted = result_df.sort_values('Distance_with_input').reset_index()
        result_df_sorted = result_df_sorted[['Image', 'Distance_with_input']]
        return result_df_sorted.head(self.topk)

    def __call__(self, input, *args, **kwargs):
        if self.embedding_name == 'vec_img_text':
            output = self.vectorize_img(input)
        elif self.embedding_name == 'vec_text':
            output = self.vectorize_img(input)
        else:
            output = self.auto_markup(input)
        return output


# class ImageTextEncoder(Model):
#     def __init__(self, config: str):
#         super().__init__(config)

