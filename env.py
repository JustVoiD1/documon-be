from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

    # All these are required strings. If missing, Pydantic throws a clean ValidationError.
    MODEL: str
    HF_TOKEN: str
    GROQ_API_KEY: str
    DATABASE_URL: str
    
    # Postgres
    PGHOST: str
    PGDATABASE: str
    PGUSER: str
    PGPASSWORD: str
    PGSSLMODE: str
    PGCHANNELBINDING: Optional[str] = None  # Explicitly marked optional (str | None)

    # AWS / S3
    AWS_ENDPOINT_URL_S3: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    S3_BUCKET: str

    PORT: int

# Instantiate immediately on import
settings = Settings()
