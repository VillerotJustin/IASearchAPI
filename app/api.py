# Import main FastAPI modules
from fastapi import FastAPI, Depends

# Internal packages
from authorization import auth
from authorization.auth import get_current_active_user
from ia import ia
from user_management import users
from graph import crud
from query import cypher
from server import test_server
from utils.environment import settings
from utils.model import loaded_model

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

