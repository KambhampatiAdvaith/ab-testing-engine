from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite+aiosqlite:///./ab_testing.db"
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    websocket_tick_ms: int = 500


settings = Settings()
