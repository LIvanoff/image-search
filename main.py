import io

from PIL import Image
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
from src.inference import ModelLauncher

app = FastAPI()


class Item(BaseModel):
    filename: str
    content: bytes


class ItemImg(BaseModel):
    content: bytes


@app.post("/tagging/")
async def tagging(item: ItemImg):
    # для проверки
    # image_path = "Radisson Collection Hotel.jpg"
    #
    # # Чтение изображения в бинарном режиме и преобразование в bytes
    # with open(image_path, "rb") as f:
    #     image_bytes = f.read()

    image = Image.open(io.BytesIO(item.content))
    model = ModelLauncher('tagging')
    tags = model.tagging(image)
    return {"tags": tags}


@app.post("/find_images/")
async def find_images(item: ItemImg):
    image = Image.open(io.BytesIO(item.content))
    model = ModelLauncher('image_text_encoding')
    #todo сюда дописать 2 параметр список всех векторов
    #todo для этого сходить в бд
    similar_images = model.find_images(image)
    return {"similar_images": similar_images}


@app.post("/vectorize/")
async def vectorize(item: ItemImg):
    image = Image.open(io.BytesIO(item.content))
    model = ModelLauncher('image_text_encoding')  # Используем ту же модель, что и для поиска изображений
    vector = model.vectorize(image)
    return {"vector": vector.tolist()}  # Преобразуем numpy array в список


@app.post("/vectorize_text/")
async def vectorize_text(text: str):
    model = ModelLauncher('text_encoding')
    vector = model.vectorize(text)
    return {"vector": vector.tolist()}  # Преобразуем numpy array в список


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
