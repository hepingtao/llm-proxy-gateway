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
        raw_aliases = self._config.get("aliases", {})
        # Separate simple string aliases from cascade dict aliases
        self._aliases: dict[str, str] = {}
        self._cascade_aliases: dict[str, dict] = {}
        for key, val in raw_aliases.items():
            if isinstance(val, dict):
                self._cascade_aliases[key] = val
            else:
                self._aliases[key] = val

    def _flatten_providers(self, providers: dict) -> dict:
        """Flatten provider-nested models into a flat dict with provider_name/model_name keys,
        inheriting provider defaults."""
        flat = {}
        for provider_name, provider_cfg in providers.items():
            provider_base_url = provider_cfg.get("base_url")
            provider_api_key_env = provider_cfg.get("api_key_env", "")
            provider_api_version = provider_cfg.get("api_version", "")
            provider_supported_formats = provider_cfg.get("supported_formats", [])

            for model_name, model_cfg in provider_cfg.get("models", {}).items():
                cfg = dict(model_cfg) if model_cfg else {}
                cfg.setdefault("upstream_model", model_name)
                cfg["provider"] = provider_name

                # Store raw base_url (may be dict or string), resolve at call time
                cfg["base_url"] = provider_base_url

                # Inherit provider-level settings if not overridden
                cfg.setdefault("api_key_env", provider_api_key_env)
                cfg.setdefault("api_version", provider_api_version)

                # Inherit supported_formats (list of format names the string URL supports)
                if provider_supported_formats:
                    cfg.setdefault("supported_formats", provider_supported_formats)

                # Key format: provider_name/model_name
                flat_key = f"{provider_name}/{model_name}"
                flat[flat_key] = cfg

        return flat

    def _resolve(self, model_name: str) -> str:
        """Resolve a model name: check simple aliases first, then return as-is."""
        return self._aliases.get(model_name, model_name)

    def is_known_model(self, model_name: str) -> bool:
        """Check if a model name resolves to an actual provider model."""
        resolved = self._resolve(model_name)
        return resolved in self._models

    def should_use_cascade(self, model_name: str) -> bool:
        """Check if a request should use cascade routing.
        True when model_name is a cascade alias OR when it's not a known model."""
        if model_name in self._cascade_aliases:
            return True
        if model_name in self._aliases:
            return not self.is_known_model(model_name)
        # Not an alias at all — if it's also not a provider model, use cascade
        return not self.is_known_model(model_name)

    def get_cascade_alias_name(self, model_name: str) -> str | None:
        """If model_name is a cascade alias, return it. Otherwise return None."""
        if model_name in self._cascade_aliases:
            return model_name
        return None

    def get_model(self, model_name: str) -> dict | None:
        name = self._resolve(model_name)
        if name in self._models:
            return self._models[name]
        if self._default:
            return self._models.get(self._default)
        return None

    def list_models(self) -> list[str]:
        names = list(self._models.keys()) + list(self._aliases.keys())
        names.extend(self._cascade_aliases.keys())
        return names

    def get_upstream_info(self, model_name: str, request_format: RequestFormat = RequestFormat.ANTHROPIC_MESSAGES) -> dict:
        cfg = self.get_model(model_name)
        if not cfg:
            raise ValueError(f"Unknown model: {model_name}")

        api_key = os.environ.get(cfg["api_key_env"], "")
        if not api_key:
            raise ValueError(f"API key not set for model {model_name} (env: {cfg['api_key_env']})")

        base_url_cfg = cfg["base_url"]
        supported_formats = cfg.get("supported_formats", [])

        if isinstance(base_url_cfg, dict):
            # Case 2: dict base_url — different URLs per format
            available_formats = [k for k, v in base_url_cfg.items() if v]
            if not available_formats:
                raise ValueError(f"Model '{model_name}' has empty base_url dict")
            if request_format.value in available_formats:
                actual_format = request_format.value
                base_url = base_url_cfg[actual_format]
            else:
                actual_format = self._pick_fallback_format(available_formats)
                if not actual_format:
                    raise ValueError(
                        f"Model '{model_name}' does not support {request_format.value} "
                        f"(supports: {', '.join(available_formats)})"
                    )
                base_url = base_url_cfg[actual_format]

        elif supported_formats:
            # Case 1: string base_url with explicit supported_formats
            base_url = base_url_cfg
            if request_format.value in supported_formats:
                actual_format = request_format.value
            else:
                actual_format = self._pick_fallback_format(supported_formats)
                if not actual_format:
                    raise ValueError(
                        f"Model '{model_name}' does not support {request_format.value} "
                        f"(supports: {', '.join(supported_formats)})"
                    )

        else:
            # No supported_formats — not allowed
            raise ValueError(
                f"Model '{model_name}' has string base_url without supported_formats. "
                "Please add supported_formats (e.g., supported_formats: [openai-chat-completions, anthropic-messages])."
            )

        return {
            "actual_format": actual_format,
            "provider_name": cfg["provider"],
            "upstream_model": cfg["upstream_model"],
            "api_key": api_key,
            "base_url": base_url,
            "api_version": cfg.get("api_version", ""),
            "reasoning_effort": cfg.get("reasoning_effort"),
            "supports_vision": cfg.get("supports_vision", False),
        }

    # Format fallback priority: openai-chat-completions > anthropic-messages > openai-responses
    _FORMAT_PRIORITY = [
        RequestFormat.OPENAI_CHAT_COMPLETIONS.value,
        RequestFormat.ANTHROPIC_MESSAGES.value,
        RequestFormat.OPENAI_RESPONSES.value,
    ]

    def _pick_fallback_format(self, available_formats: list[str]) -> str | None:
        """Pick the best fallback format from available formats, following priority order.
        Returns the format string or None if no match."""
        for fmt in self._FORMAT_PRIORITY:
            if fmt in available_formats:
                return fmt
        return None

    @property
    def log_dir(self) -> Path:
        return CONFIG_DIR / self._config.get("log_dir", "logs")

    def get_cascade_config(self, alias_name: str = "auto") -> dict:
        """Return the raw cascade config block for the given alias name (default 'auto')."""
        return self._cascade_aliases.get(alias_name, {})

    def get_cascade_upstream_list(self, request_format: RequestFormat, alias_name: str = "auto") -> list[dict]:
        """
        Return a list of upstream info dicts in cascade order for the given alias.
        Skips models that fail to resolve (e.g., missing API key) with a warning.
        Raises ValueError if no valid models found.
        """
        alias_cfg = self._cascade_aliases.get(alias_name, {})
        if not alias_cfg or "cascade" not in alias_cfg:
            raise ValueError(f"No '{alias_name}' cascade configuration found in config.yaml")

        # Alias-level default for max_output_tokens (used when the client
        # request omits max_tokens / max_output_tokens). Injected into each
        # upstream dict so request-conversion code can pick it up.
        alias_default_max_tokens = alias_cfg.get("default_max_output_tokens")

        upstream_list = []
        for entry in alias_cfg["cascade"]:
            model_name = entry["model"]
            try:
                upstream = self.get_upstream_info(model_name, request_format)
                upstream["_cascade_model_name"] = model_name
                if alias_default_max_tokens is not None:
                    upstream["default_max_output_tokens"] = alias_default_max_tokens
                upstream_list.append(upstream)
            except ValueError as e:
                print(f"[config] WARNING: Skipping cascade model '{model_name}': {e}")
                continue

        if not upstream_list:
            raise ValueError(f"No valid models in '{alias_name}' cascade configuration")
        return upstream_list


@lru_cache
def get_settings() -> Settings:
    return Settings()


_config_path = CONFIG_DIR / "config.yaml"
_config_mtime: float = 0
_config_instance: ModelConfig | None = None


def get_model_config() -> ModelConfig:
    """Get ModelConfig with automatic hot-reload when config.yaml changes (by mtime)."""
    global _config_mtime, _config_instance
    try:
        current_mtime = _config_path.stat().st_mtime
    except OSError:
        current_mtime = 0
    if _config_instance is None or current_mtime > _config_mtime:
        _config_mtime = current_mtime
        _config_instance = ModelConfig()
    return _config_instance
