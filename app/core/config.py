"""Central settings. Never read os.getenv outside this module."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    groq_model_features: str = "llama-3.3-70b-versatile"
    groq_model_jev: str = "llama-3.1-8b-instant"
    jev_batch_size: int = 10
    jev_max_concurrency: int = 5

    scraper_headless: bool = True
    scraper_min_delay_s: float = 1.0
    scraper_max_delay_s: float = 2.5
    max_review_pages: int = 100

    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
