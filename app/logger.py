import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import get_model_config


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name)


class ConversationLogger:
    def __init__(self):
        self._log_dir = get_model_config().log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _today_file(self, downstream_model: str, upstream_model: str) -> Path:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        name = f"conversations_{_safe_name(downstream_model)}_{_safe_name(upstream_model)}_{today}.json"
        return self._log_dir / name

    def _read_log(self, path: Path) -> list:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                # Log file corrupted, start fresh
                return []
        return []

    def _write_log(self, path: Path, data: list):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def log_request(
        self, downstream_model: str, upstream_model: str,
        messages: list[dict], client_id: str = "",
        extra_params: dict | None = None,
        base_url: str = "",
    ) -> str:
        conv_id = str(uuid.uuid4())
        entry = {
            "id": conv_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "client_id": client_id,
            "downstream_model": downstream_model,
            "upstream_model": upstream_model,
            "base_url": base_url,
            "messages": messages,
            "extra_params": extra_params or {},
            "response": None,
        }
        path = self._today_file(downstream_model, upstream_model)
        logs = self._read_log(path)
        logs.append(entry)
        self._write_log(path, logs)
        return conv_id

    def log_response(self, downstream_model: str, upstream_model: str, conv_id: str, response_text: str, usage: dict, finish_reason: str):
        path = self._today_file(downstream_model, upstream_model)
        logs = self._read_log(path)
        for entry in logs:
            if entry["id"] == conv_id:
                entry["response"] = {
                    "text": response_text,
                    "usage": usage,
                    "finish_reason": finish_reason,
                    "request_timestamp": entry["timestamp"],
                    "response_timestamp": datetime.now(timezone.utc).isoformat(),
                }
                break
        self._write_log(path, logs)


logger = ConversationLogger()
