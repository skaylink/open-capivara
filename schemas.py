import pydantic
from typing import Optional, Dict, List


class Signature(pydantic.BaseModel):
    name: Optional[str]
    version: str = '0.0.1'
    tags: Dict[str, int] = {}
    embedding: List[float] = []

class Entity(pydantic.BaseModel):
    name: Optional[str]
    signature: Optional[Signature]

class EntityText(Entity):
    name: Optional[str]
    text: str
    signature: Optional[Signature]

class EntityTicket(Entity):
    pass

class EntityPerson(Entity):
    pass