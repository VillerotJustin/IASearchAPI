# main.py
from typing import ClassVar, Optional, List

from fastapi import FastAPI, HTTPException
from neontology import BaseNode, BaseRelationship, init_neontology

from Models import Team, TeamMember
from Routers import Administration

class BelongsTo(BaseRelationship):
    __relationshiptype__: ClassVar[str] = "BELONGS_TO"
    source: TeamMember
    target: Team


from config import settings

app = FastAPI()

app.include_router(Administration.router)
