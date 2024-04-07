import io
from typing import Optional

from PIL import Image
from fastapi import APIRouter, UploadFile, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import database
from src.models.inference import ModelLauncher
from src.photos.models import Photo
from src.photos import service as photos_service

router = APIRouter()


class Item(BaseModel):
    filename: str
    content: bytes


class ItemImg(BaseModel):
    content: bytes


@router.post("/find_images/")
async def find_images(
        filename: str | None = None,
        file: UploadFile | None = None,
        text: Optional[str] = None,
        filters: dict | None = None,
        session: AsyncSession = Depends(database.session_dependency),
):
    photos: list[Photo] = await photos_service.get_photos(
        session=session
    )
    if text is not None:
        input = text
        model = ModelLauncher("text_encoding")
    else:
        input = Image.open(file.file)
        model = ModelLauncher("image_text_encoding")

    image_db = model.create_images_df(photos, filters)
    similar_images = model.find_images(input=input, images_db=image_db)
    return {"similar_images": similar_images.id.tolist()}


@router.post("/vectorize_image/")
async def vectorize(item: ItemImg):
    image = Image.open(io.BytesIO(item.content))
    model = ModelLauncher(
        "image_text_encoding"
    )  # Используем ту же модель, что и для поиска изображений
    vector = model.vectorize(image)
    return {"vector": vector.tolist()}  # Преобразуем numpy array в список


@router.post("/vectorize_text/")
async def vectorize_text(text: str):
    model = ModelLauncher("text_encoding")
    vector = model.vectorize(text)
    return {"vector": vector.tolist()}  # Преобразуем numpy array в список


@router.post("/all")
async def all(filename: str, content: UploadFile):
    model_text = ModelLauncher("text_encoding")
    model_img = ModelLauncher("image_text_encoding")
    model_tags = ModelLauncher("tagging")

    image = Image.open(content.file)

    vector_text = model_text.vectorize(filename)
    vector_img = model_img.vectorize(image)
    tags = model_tags.tagging(image)

    return {
        "vector_text": vector_text.tolist(),
        "vector_img": vector_img.tolist(),
        "tags": tags,
    }
