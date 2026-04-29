import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from .models import RequestFormat

load_dotenv()

CONFIG_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 4936

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class ModelConfig:
    def __init__(self):
        config_path = CONFIG_DIR / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)
        self._models = self._flatten_providers(self._config.get("providers", {}))
        self._default = self._config.get("default_model", "")
        self._aliases = self._config.get("aliases", {})

    def _flatten_providers(self, providers: dict) -> dict:
        """Flatten provider-nested models into a flat dict, inheriting provider defaults."""
        flat = {}
        for provider_name, provider_cfg in providers.items():
            provider_base_url = provider_cfg.get("base_url")
            provider_api_key_env = provider_cfg.get("api_key_env", "")
            provider_api_version = provider_cfg.get("api_version", "")

            for model_name, model_cfg in provider_cfg.get("models", {}).items():
                cfg = dict(model_cfg) if model_cfg else {}
                cfg.setdefault("upstream_model", model_name)
                cfg["provider"] = provider_name

                # Store raw base_url (may be dict or string), resolve at call time
                cfg["base_url"] = provider_base_url

                # Inherit provider-level settings if not overridden
                cfg.setdefault("api_key_env", provider_api_key_env)
                cfg.setdefault("api_version", provider_api_version)

                flat[model_name] = cfg

        return flat

    def _resolve(self, model_name: str) -> str:
        return self._aliases.get(model_name, model_name)

    def get_model(self, model_name: str) -> dict | None:
        name = self._resolve(model_name)
        if name in self._models:
            return self._models[name]
        if self._default:
            return self._models.get(self._default)
        return None

    def list_models(self) -> list[str]:
        return list(self._models.keys()) + list(self._aliases.keys())

    def get_upstream_info(self, model_name: str, request_format: RequestFormat = RequestFormat.ANTHROPIC) -> dict:
        cfg = self.get_model(model_name)
        if not cfg:
            raise ValueError(f"Unknown model: {model_name}")

        api_key = os.environ.get(cfg["api_key_env"], "")
        if not api_key:
            raise ValueError(f"API key not set for model {model_name} (env: {cfg['api_key_env']})")

        base_url = cfg["base_url"]
        # For multi-URL providers, resolve based on request format
        if isinstance(base_url, dict):
            base_url = base_url.get(request_format, "")

        return {
            "provider": cfg["provider"],
            "upstream_model": cfg["upstream_model"],
            "api_key": api_key,
            "base_url": base_url,
            "api_version": cfg.get("api_version", ""),
            "reasoning_effort": cfg.get("reasoning_effort"),
        }

    @property
    def log_dir(self) -> Path:
        return CONFIG_DIR / self._config.get("log_dir", "logs")


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_model_config() -> ModelConfig:
    return ModelConfig()
