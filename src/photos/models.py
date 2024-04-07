from datetime import datetime
from typing import Optional

from sqlalchemy import Float, ForeignKey
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.model import Base


class Identical(Base):
    __tablename__ = "identical_photo"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    photo_left: Mapped[int] = mapped_column(ForeignKey("photo.id"))
    photo_right: Mapped[int] = mapped_column()


class Tag(Base):
    __tablename__ = "tag"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(nullable=False, unique=True)

    photos: Mapped[list["Photo"]] = relationship(
        secondary="photo_tag",
        back_populates="tags",
    )


class PhotosTags(Base):
    __tablename__ = "photo_tag"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    tag_id: Mapped[int] = mapped_column(ForeignKey("tag.id"))
    photo_id: Mapped[int] = mapped_column(ForeignKey("photo.id"))


class Photo(Base):
    __tablename__ = "photo"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    name: Mapped[str] = mapped_column(nullable=False)

    tags: Mapped[list["Tag"]] = relationship(
        secondary="photo_tag",
        back_populates="photos",
    )

    similar_photos: Mapped[list["Identical"]] = relationship()

    has_people: Mapped[bool] = mapped_column(nullable=False)

    primary_color: Mapped[int] = mapped_column(nullable=True)

    hash: Mapped[str] = mapped_column(nullable=False)
    status: Mapped[int] = mapped_column(nullable=False)

    text_vector: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=True)
    image_vector: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=True)

    url: Mapped[str] = mapped_column(nullable=False, unique=True)

    season: Mapped[int] = mapped_column(nullable=True)
    day_time: Mapped[int] = mapped_column(nullable=True)
    orientation: Mapped[int] = mapped_column(nullable=True)
    format: Mapped[int] = mapped_column(nullable=True)

    longitude: Mapped[float] = mapped_column(nullable=True)
    latitude: Mapped[float] = mapped_column(nullable=True)

    slag: Mapped[str] = mapped_column(nullable=False, unique=True)

    height: Mapped[int] = mapped_column(nullable=False)
    width: Mapped[int] = mapped_column(nullable=False)

    file_size_name: Mapped[int] = mapped_column(nullable=False)

    views: Mapped[int] = mapped_column(nullable=False, default=0)
    download_amount: Mapped[int] = mapped_column(nullable=False, default=0)
    rating: Mapped[float] = mapped_column(nullable=False, default=5.0)

    created_at: Mapped[datetime] = mapped_column(
        nullable=False, default=datetime.utcnow
    )
    created_by: Mapped[str] = mapped_column(nullable=False)

    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, default=datetime.utcnow
    )
    updated_by: Mapped[str] = mapped_column(nullable=False)


# class UserFilters(Base):
#     season: Optional[int] = None
#     day_time: Optional[int] = None
#     orientation: Optional[int] = None
#     format: Optional[int] = None
#     file_size_name: Optional[int] = None
#     has_people: Optional[bool] = None
#     primary_color: Optional[int] = None


# class Admin(UserFilters):
#     status: Optional[int] = None
