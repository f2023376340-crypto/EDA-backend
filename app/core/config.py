"""Central settings. Never read os.getenv outside this module."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    groq_model_features: str = "openai/gpt-oss-120b"

    jev_api_key: str = ""
    jev_base_url: str = "https://api.typesafe.ai/v1"
    jev_model: str = "jev-1.14.0"
    jev_max_concurrency: int = 10

    scraper_headless: bool = True
    scraper_min_delay_s: float = 1.0
    scraper_max_delay_s: float = 2.5
    max_review_pages: int = 100

    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
