from fastapi import FastAPI

from src.config import settings
from src.models.router import router as model_router

app = FastAPI(
    title=settings.app.title,
    root_path=settings.app.prefix,
)

app.include_router(
    router=model_router,
    prefix="/model",
    tags=["Model"],
)
