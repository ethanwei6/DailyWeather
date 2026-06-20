from __future__ import annotations

import json
from typing import Any, Mapping, Optional
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class HttpClient:
    def __init__(self, user_agent: str = "weather-polymarket-strategy/0.1", timeout_seconds: int = 5):
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds

    def get_json(self, url: str, params: Optional[Mapping[str, Any]] = None, headers: Optional[Mapping[str, str]] = None) -> Any:
        return json.loads(self.get_text(url, params=params, headers=headers))

    def get_text(self, url: str, params: Optional[Mapping[str, Any]] = None, headers: Optional[Mapping[str, str]] = None) -> str:
        query = urlencode({key: value for key, value in (params or {}).items() if value is not None}, doseq=True)
        full_url = f"{url}?{query}" if query else url
        request_headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        request_headers.update(headers or {})
        request = Request(full_url, headers=request_headers)
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read().decode("utf-8")
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {error.code} for {full_url}: {body[:500]}") from error
        except (URLError, TimeoutError, OSError) as error:
            raise RuntimeError(f"Request failed for {full_url}: {error}") from error
        return payload
