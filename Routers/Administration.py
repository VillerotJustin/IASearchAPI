from neontology import init_neontology
from fastapi import APIRouter, Depends
from ..dependencies import get_token_header
from ..config import settings

router = APIRouter(
    prefix="/",
    tags=["Admin"],
    dependencies=[Depends(get_token_header)],
    responses={404: {"description": "Not found"}},
)


# Neo4j database connection
@router.on_event("startup")
async def startup_event():
    init_neontology(
        init_neontology(
            neo4j_uri=settings.NEO4J_URI,
            neo4j_username=settings.NEO4J_USERNAME,
            neo4j_password=settings.NEO4J_PASSWORD,
        )
    )


@router.get("/info")
def info():
    return {
        "app_name": settings.app_name,
        "admin_email": settings.admin_email,
        "items_per_user": settings.items_per_user,
    }


@router.get("/neo4j")
def read_root():
    return {"foo": "bar"}


@router.get("/up")
def read_root():
    return {"message": "the server is up"}
