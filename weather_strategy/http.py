from __future__ import annotations

import json
from typing import Any, Mapping, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen


_SENSITIVE_PARAM_MARKERS = ("api_key", "apikey", "key", "token", "secret", "password", "authorization")


class HttpClient:
    def __init__(self, user_agent: str = "weather-polymarket-strategy/0.1", timeout_seconds: int = 5):
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds

    def get_json(self, url: str, params: Optional[Mapping[str, Any]] = None, headers: Optional[Mapping[str, str]] = None) -> Any:
        return json.loads(self.get_text(url, params=params, headers=headers))

    def get_text(self, url: str, params: Optional[Mapping[str, Any]] = None, headers: Optional[Mapping[str, str]] = None) -> str:
        return self.get_bytes(url, params=params, headers=headers).decode("utf-8")

    def get_bytes(self, url: str, params: Optional[Mapping[str, Any]] = None, headers: Optional[Mapping[str, str]] = None) -> bytes:
        full_url = _full_url(url, params)
        request_headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        request_headers.update(headers or {})
        request = Request(full_url, headers=request_headers)
        redacted_url = _redact_url(full_url)
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read()
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {error.code} for {redacted_url}: {body[:500]}") from error
        except (URLError, TimeoutError, OSError) as error:
            raise RuntimeError(f"Request failed for {redacted_url}: {error}") from error
        return payload

    def get_redirect_location(self, url: str, params: Optional[Mapping[str, Any]] = None, headers: Optional[Mapping[str, str]] = None) -> Optional[str]:
        full_url = _full_url(url, params)
        request_headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        request_headers.update(headers or {})
        request = Request(full_url, headers=request_headers)
        redacted_url = _redact_url(full_url)
        opener = build_opener(_NoRedirectHandler)
        try:
            with opener.open(request, timeout=self.timeout_seconds) as response:
                location = response.headers.get("Location")
                return location
        except HTTPError as error:
            if 300 <= error.code < 400:
                location = error.headers.get("Location")
                if location:
                    return location
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {error.code} for {redacted_url}: {body[:500]}") from error
        except (URLError, TimeoutError, OSError) as error:
            raise RuntimeError(f"Request failed for {redacted_url}: {error}") from error


def _full_url(url: str, params: Optional[Mapping[str, Any]] = None) -> str:
    query = urlencode({key: value for key, value in (params or {}).items() if value is not None}, doseq=True)
    return f"{url}?{query}" if query else url


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


def _redact_url(url: str) -> str:
    parts = urlsplit(url)
    if not parts.query:
        return url
    redacted_query = urlencode(
        [
            (key, "REDACTED" if _is_sensitive_param(key) else value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
        ]
    )
    return urlunsplit((parts.scheme, parts.netloc, parts.path, redacted_query, parts.fragment))


def _is_sensitive_param(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(marker in normalized for marker in _SENSITIVE_PARAM_MARKERS)
