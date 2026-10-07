from pydantic import BaseModel


class Item(BaseModel):
    description: str | None = None

class Category(BaseModel):
    category: str | None = None