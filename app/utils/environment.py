# Packages and functions for loading environment variables
import os
from dotenv import load_dotenv, find_dotenv

# Load environment from disk first, then apply any defaults
load_dotenv(find_dotenv('../../.env'))


class Config:
    # App description
    APP_NAME = os.environ.get('APP_NAME', 'IASearchAPI')
    APP_DESC = os.environ.get('APP_DESC', '')
    APP_VERSION = os.environ.get('APP_VERSION', '')

    # Documentation location
    DOCS_URL = os.environ.get('DOCS_URL', '/docs')
    REDOC_URL = os.environ.get('REDOC_URL', '/redoc')

    # NEO4J DB Info
    NEO4J_URI = os.environ.get('NEO4J_URI', '')
    NEO4J_USERNAME = os.environ.get('NEO4J_USERNAME', 'neo4j')
    NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', '')
    DB_PREFIX = os.environ.get('DB_PREFIX', '')

    # Application password for superadmin functions (/auth/launch_user endpoint for first-time setup)
    APP_PASSWORD = os.environ.get('APP_PASSWORD')

    # Settings for encryption
    SECRET_KEY = os.environ.get('SECRET_KEY', 'secret')
    ALGORITHM = os.environ.get('ALGORITHM', "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get('ACCESS_TOKEN_EXPIRE_MINUTES', 10_080))  # one week

    if os.environ.get('LOAD_LOCAL_MODEL', '') == 'true':
        LOAD_LOCAL_MODEL = True
    else:
        LOAD_LOCAL_MODEL = False


settings = Config()
