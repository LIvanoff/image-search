import io
from typing import Optional

from PIL import Image
from fastapi import APIRouter, UploadFile, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import numpy as np

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


@router.post("/find_images")
async def find_images(
    text: Optional[str] = None,
    filters: dict | None = None,
    session: AsyncSession = Depends(database.session_dependency),
):
    photos: list[Photo] = await photos_service.get_photos(session=session)
    input = text
    model = ModelLauncher("text_encoding")

    image_db = model.create_images_df(photos, filters)
    similar_images = model.find_images(input=input, images_db=image_db)
    return {"similar_images": similar_images.id.tolist()}


@router.post("/find_images_photo")
async def find_images(
    file: UploadFile,
    session: AsyncSession = Depends(database.session_dependency),
):
    photos: list[Photo] = await photos_service.get_photos(session=session)

    input = Image.open(file.file)
    model = ModelLauncher("image_text_encoding")

    image_db = model.create_images_df(photos)
    similar_images = model.find_images(input=input, images_db=image_db)
    return {"similar_images": similar_images.id.tolist()}


async def search_photos(
        filters: dict,
        session: AsyncSession,
        text: Optional[str] | None = None,
        file: Optional[UploadFile] | None = None,
):
    query = select(Photo)
    filters_keys = list(filters.keys())
    if 'season' in filters_keys:
        query = query.where(Photo.season == filters['season'])
    if 'day_time' in filters_keys:
        query = query.where(Photo.day_time == filters['day_time'])
    if 'orientation' in filters_keys:
        query = query.where(Photo.orientation == filters['orientation'])
    if 'format' in filters_keys:
        query = query.where(Photo.format == filters['format'])
    if 'file_size_name' in filters_keys:
        query = query.where(Photo.file_size_name == filters['file_size_name'])
    if 'has_people' in filters_keys:
        query = query.where(Photo.has_people == filters['has_people'])
    if 'primary_color' in filters_keys:
        query = query.where(Photo.primary_color == filters['primary_color'])
    if 'status' in filters_keys:
        query = query.where(Photo.status == filters['status'])
    result = await session.scalars(query)
    return result


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

    palette = get_palette(image)
    # print(palette)

    return {
        "vector_text": vector_text.tolist(),
        "vector_img": vector_img.tolist(),
        "tags": tags,
        "palette": palette.tolist()
    }


def get_palette(image, palette_size: int = 5):
    arr = np.asarray(image)
    palette, index = np.unique(asvoid(arr).ravel(), return_inverse=True)
    palette = palette.view(arr.dtype).reshape(-1, arr.shape[-1])
    count = np.bincount(index)
    order = np.argsort(count)
    return palette[order[::-1]][:palette_size]


def asvoid(arr):
    arr = np.ascontiguousarray(arr)
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))
