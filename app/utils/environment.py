from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    # App description
    APP_NAME: str = "IASearchAPI"
    APP_DESC: str
    APP_VERSION: str

    # Documentation location
    DOCS_URL: str = "/docs"
    REDOC_URL: str = "/redoc"

    # NEO4J DB Info
    NEO4J_URI: str
    NEO4J_USERNAME: str
    NEO4J_PASSWORD: str

    # Application password for superadmin functions (/auth/launch_user endpoint for first-time setup)
    APP_PASSWORD: str

    # Settings for encryption
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10_080

    model_config = SettingsConfigDict(env_file="../.env")


settings = Config()
