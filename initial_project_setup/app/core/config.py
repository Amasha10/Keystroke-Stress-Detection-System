from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PAUSE_THRESHOLD_S: float = 1.0
    LABEL_TOP_QUANTILE: float = 0.7

settings = Settings()
