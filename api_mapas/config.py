from pydantic import BaseSettings

class Settings(BaseSettings):
    DB_URL: str
    DB_NAME: str
    HOST: str = "127.0.0.1"
    PORT: int = 9000
    DEBUG_MODE: bool = True

    class Config:
        env_file = ".env"  # ← aquí indica el archivo .env

settings = Settings()  # ← instancia que usarás en main.py
