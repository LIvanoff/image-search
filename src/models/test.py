from sentence_transformers import SentenceTransformer
from scipy import spatial
from PIL import Image
import pandas as pd
import numpy as np
import copy
import os

model_name = 'clip-ViT-B-16'
st_model = SentenceTransformer(model_name)


def vectorize_img(img_path: str, search: bool = False, model: SentenceTransformer = st_model) -> np.array:
    if not search:
        img = Image.open(img_path)
    else:
        img = img_path
    return st_model.encode(img)


def create_images_db(images_folder: str, model: SentenceTransformer = st_model) -> pd.DataFrame:
    data_dict = dict()
    for file_name in os.listdir(images_folder):
        image_path = os.path.join(images_folder, file_name)
        if os.path.isfile(image_path):
            emb = vectorize_img(image_path)
            data_dict[file_name] = emb
    images_db = pd.DataFrame(data_dict.items(), columns=['Image', 'Embedding'])
    images_db.to_excel(f"{name}.xlsx")
    return images_db


def get_df(df_path: str) -> pd.DataFrame:
    data_df = pd.read_json(df_path)
    data_df['Embedding'] = data_df['Embedding'].apply(lambda x: np.array(x))
    return data_df


def calculate_cos_dist(emb_a: np.array, emb_b: np.array) -> float:
    result_distance = spatial.distance.cosine(emb_a, emb_b)
    return result_distance


def found_similar_images(input_img_path: str, images_db: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    input_vec = vectorize_img(input_img_path, search=True)
    result_df = copy.deepcopy(images_db)
    result_df['Distance_with_input'] = result_df.apply(lambda x: calculate_cos_dist(input_vec, x['Embedding']), axis=1)
    result_df_sorted = result_df.sort_values('Distance_with_input').reset_index()
    result_df_sorted = result_df_sorted[['Image', 'Distance_with_input']]
    return result_df_sorted.head(n)


name = 'moscow'
images_folder = f'D:/MoscowTravelHack/{name}'
images_db = create_images_db(images_folder)

print(images_db)
# input_img_path = f'D:/MoscowTravelHack/{name}/Новодевичий монастырь_7.jpg'
input_img_path = 'Новодевичий монастырь'
result_df = found_similar_images(input_img_path, images_db)
print(result_df)
