# Lib
import time

from fastapi import APIRouter
from graphdatascience import GraphDataScience

# Internal
from utils.environment import settings
from utils.db import neo4j_driver

# Set the API Router
router = APIRouter()


@router.get("/info")
def info():
    return {
        "app_name": settings.APP_NAME,
    }


@router.get("/neo4j")
async def check_DB():
    try:
        server_info = neo4j_driver.get_server_info()
        print('Connection established')
        print(server_info)
        return {"message": f"Connection established : {server_info}"}
    except Exception as e:
        print(f"Connection error : {e}")
        return { "message": f"Connection error : {e}"}


@router.get("/up")
def is_up():
    return {"message": "the server is up"}
