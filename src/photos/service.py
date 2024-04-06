from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.photos.models import Photo


async def get_photos(session: AsyncSession):
    result = await session.scalars(select(Photo))
    return result.all()
