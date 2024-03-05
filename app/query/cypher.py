# Import required base modules
from dotenv import load_dotenv, find_dotenv

# Import modules from FastAPI
from fastapi import APIRouter, HTTPException
from starlette import status

# Import internal utilities for database access and schemas
from app.utils.db import neo4j_driver
from app.utils.schema import Query

# Load environment variables
env_loc = find_dotenv('.env')
load_dotenv(env_loc)

# Set the API Router
router = APIRouter()


# Query endpoint
@router.post('/q', response_model=Query, summary='Query the database with a custom Cypher string')
async def cypher_query(attributes: dict):
    print(attributes)
    # print(attributes["cypher_string"])
    if attributes["cypher_string"] is not None and attributes["cypher_string"] != "":
        with neo4j_driver.session() as session:
            response = session.run(query=attributes["cypher_string"])
            data = response.data()
            print(data)
            return Query(response=data)
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Empty or null cypher string.",
            headers={"WWW-Authenticate": "Bearer"})