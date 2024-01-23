# Import main FastAPI modules
from fastapi import FastAPI, Depends

# Internal packages
from app.authorisation import auth
from app.authorisation.auth import get_current_active_user
from app.user_management import users
from app.graph import crud
from app.query import cypher
from app.server import test_server
from app.utils.environment import settings


app = FastAPI(
    title=settings.app_name,
    description=settings.app_desc,
    version=settings.app_version,
    docs_url=settings.docs_url,
    redoc_url=settings.redoc_url
)

app.include_router(
    test_server.router,
    prefix='/test',
    tags=['test']
)

app.include_router(
    auth.router,
    prefix='/auth',
    tags=['Authorisation']
)

app.include_router(
    users.router,
    prefix='/users',
    tags=['Users'],
    dependencies=[Depends(get_current_active_user)]
)

app.include_router(
    crud.router,
    prefix='/graph',
    tags=['Graph Objects'],
    dependencies=[Depends(get_current_active_user)]
)

app.include_router(
    cypher.router,
    tags=['Query Database'],
    dependencies=[Depends(get_current_active_user)]
)