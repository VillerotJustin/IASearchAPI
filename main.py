# main.py
from typing import ClassVar, Optional, List

from fastapi import FastAPI, HTTPException
from neontology import BaseNode, BaseRelationship, init_neontology

from Models import Team, TeamMember


class BelongsTo(BaseRelationship):
    __relationshiptype__: ClassVar[str] = "BELONGS_TO"
    source: TeamMember
    target: Team

# .env
import os

name = os.getenv("MY_NAME", "World")
print(f"Hello {name} from Python")


app = FastAPI()

from Routers import Administration

app.include_router(Administration.router)