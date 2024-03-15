# Import main FastAPI modules
from fastapi import FastAPI, Depends

# Internal packages
from app.authorisation import auth
from app.authorisation.auth import get_current_active_user
from app.ia import ia
from app.user_management import users
from app.graph import crud
from app.query import cypher
from app.server import test_server
from app.utils.environment import settings
from app.utils.model import loaded_model

frWac_model = loaded_model


######################
# Fast API
######################
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESC,
    version=settings.APP_VERSION,
    docs_url=settings.DOCS_URL,
    redoc_url=settings.REDOC_URL
)

#####################
# Endpoints
#####################
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

app.include_router(
    ia.router,
    prefix='/IA',
    tags=['IA search'],
    dependencies=[Depends(get_current_active_user)]
)

