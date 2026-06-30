from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, Iterable, Optional

from weather_strategy.http import HttpClient


DEFAULT_ENV_PATH = ".env.local"
DEFAULT_CLOB_API_URL = "https://clob.polymarket.com"
DEFAULT_CHAIN_ID = "137"

_SECRET_KEYS = {
    "PRIVATE_KEY",
    "CLOB_SECRET",
    "CLOB_PASS_PHRASE",
    "BUILDER_SECRET",
    "BUILDER_PASS_PHRASE",
}


class LiveSetupError(RuntimeError):
    pass


def create_bot_wallet(env_file: str = DEFAULT_ENV_PATH, overwrite: bool = False) -> dict[str, Any]:
    env_path = Path(env_file)
    env = load_env_file(env_path)
    if not overwrite and env.get("PRIVATE_KEY"):
        return {
            "created": False,
            "env_file": str(env_path),
            "bot_owner_address": env.get("BOT_OWNER_ADDRESS"),
            "reason": "PRIVATE_KEY already present",
        }

    try:
        from eth_account import Account
    except ImportError as exc:
        raise LiveSetupError("eth-account is required. Install it in the live virtualenv first.") from exc

    account = Account.create()
    private_key = account.key.hex()
    if not private_key.startswith("0x"):
        private_key = "0x" + private_key

    updates = {
        "POLYMARKET_VENUE": "international",
        "CLOB_API_URL": DEFAULT_CLOB_API_URL,
        "CHAIN_ID": DEFAULT_CHAIN_ID,
        "PRIVATE_KEY": private_key,
        "BOT_OWNER_ADDRESS": account.address,
        "POLYMARKET_SIGNATURE_TYPE": "3",
        "DAILYWEATHER_LIVE_TRADING": "0",
        "DAILYWEATHER_MAX_BANKROLL_USD": "100",
        "DAILYWEATHER_MAX_ORDER_USD": "5",
        "DAILYWEATHER_MAX_DAILY_LOSS_USD": "10",
    }
    write_env_updates(env_path, updates, overwrite=overwrite)
    return {
        "created": True,
        "env_file": str(env_path),
        "bot_owner_address": account.address,
        "private_key_stored": True,
    }


def derive_clob_credentials(env_file: str = DEFAULT_ENV_PATH, overwrite: bool = False) -> dict[str, Any]:
    env_path = Path(env_file)
    env = load_env_file(env_path)
    _require_env(env, ["PRIVATE_KEY"])
    if not overwrite and all(env.get(key) for key in ("CLOB_API_KEY", "CLOB_SECRET", "CLOB_PASS_PHRASE")):
        return {
            "created": False,
            "env_file": str(env_path),
            "clob_api_key": _redact_value(env.get("CLOB_API_KEY")),
            "reason": "CLOB credentials already present",
        }

    try:
        from py_clob_client_v2 import ClobClient
    except ImportError as exc:
        raise LiveSetupError("py-clob-client-v2 is required. Install it in the live virtualenv first.") from exc

    client = ClobClient(
        host=env.get("CLOB_API_URL") or DEFAULT_CLOB_API_URL,
        chain_id=int(env.get("CHAIN_ID") or DEFAULT_CHAIN_ID),
        key=env["PRIVATE_KEY"],
    )
    credentials = client.create_or_derive_api_key()
    api_key = _credential_value(credentials, "api_key", "apiKey", "key")
    api_secret = _credential_value(credentials, "api_secret", "secret")
    api_passphrase = _credential_value(credentials, "api_passphrase", "passphrase")
    write_env_updates(
        env_path,
        {
            "CLOB_API_KEY": api_key,
            "CLOB_SECRET": api_secret,
            "CLOB_PASS_PHRASE": api_passphrase,
        },
        overwrite=overwrite,
    )
    return {
        "created": True,
        "env_file": str(env_path),
        "clob_api_key": _redact_value(api_key),
    }


def check_geoblock() -> dict[str, Any]:
    data = HttpClient(timeout_seconds=10).get_json("https://polymarket.com/api/geoblock")
    if not isinstance(data, dict):
        raise LiveSetupError(f"Unexpected geoblock response: {data!r}")
    return {
        "blocked": bool(data.get("blocked")),
        "country": data.get("country"),
        "region": data.get("region"),
        "ip": _redact_ip(data.get("ip")),
        "raw": {key: value for key, value in data.items() if key != "ip"},
    }


def live_status(env_file: str = DEFAULT_ENV_PATH) -> dict[str, Any]:
    env_path = Path(env_file)
    env = load_env_file(env_path)
    return {
        "env_file": str(env_path),
        "env_exists": env_path.exists(),
        "env_mode": _env_mode(env_path),
        "venue": env.get("POLYMARKET_VENUE"),
        "chain_id": env.get("CHAIN_ID"),
        "clob_api_url": env.get("CLOB_API_URL"),
        "bot_owner_address": env.get("BOT_OWNER_ADDRESS"),
        "deposit_wallet_address": env.get("DEPOSIT_WALLET_ADDRESS"),
        "has_private_key": bool(env.get("PRIVATE_KEY")),
        "has_clob_credentials": all(env.get(key) for key in ("CLOB_API_KEY", "CLOB_SECRET", "CLOB_PASS_PHRASE")),
        "has_builder_relayer_credentials": all(
            env.get(key) for key in ("RELAYER_URL", "BUILDER_API_KEY", "BUILDER_SECRET", "BUILDER_PASS_PHRASE")
        ),
        "live_trading_enabled": env.get("DAILYWEATHER_LIVE_TRADING") == "1",
        "max_bankroll_usd": env.get("DAILYWEATHER_MAX_BANKROLL_USD"),
        "max_order_usd": env.get("DAILYWEATHER_MAX_ORDER_USD"),
        "max_daily_loss_usd": env.get("DAILYWEATHER_MAX_DAILY_LOSS_USD"),
    }


def clob_readonly_smoke(env_file: str = DEFAULT_ENV_PATH) -> dict[str, Any]:
    env = load_env_file(Path(env_file))
    _require_env(env, ["PRIVATE_KEY", "CLOB_API_KEY", "CLOB_SECRET", "CLOB_PASS_PHRASE"])
    try:
        from py_clob_client_v2 import ApiCreds, ClobClient
    except ImportError as exc:
        raise LiveSetupError("py-clob-client-v2 is required. Install it in the live virtualenv first.") from exc

    creds = ApiCreds(
        api_key=env["CLOB_API_KEY"],
        api_secret=env["CLOB_SECRET"],
        api_passphrase=env["CLOB_PASS_PHRASE"],
    )
    client = ClobClient(
        host=env.get("CLOB_API_URL") or DEFAULT_CLOB_API_URL,
        chain_id=int(env.get("CHAIN_ID") or DEFAULT_CHAIN_ID),
        key=env["PRIVATE_KEY"],
        creds=creds,
    )
    open_orders = client.get_open_orders()
    trades = client.get_trades()
    return {
        "open_orders_count": len(open_orders) if isinstance(open_orders, list) else None,
        "trades_count": len(trades) if isinstance(trades, list) else None,
        "open_orders_response_type": type(open_orders).__name__,
        "trades_response_type": type(trades).__name__,
    }


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        values[key] = value
    return values


def write_env_updates(path: Path, updates: dict[str, str], overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = load_env_file(path)
    lines = path.read_text().splitlines() if path.exists() else []
    rendered = list(lines)
    present_keys = _line_key_indexes(rendered)
    for key, value in updates.items():
        if key in present_keys:
            if not overwrite:
                continue
            rendered[present_keys[key]] = f"{key}={value}"
        else:
            rendered.append(f"{key}={value}")
    if rendered and rendered[-1] != "":
        text = "\n".join(rendered) + "\n"
    else:
        text = "\n".join(rendered)
    path.write_text(text)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    for key in updates:
        current[key] = updates[key]


def redact_env_status(status: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(status)
    for key in list(redacted):
        if key.upper() in _SECRET_KEYS:
            redacted[key] = "REDACTED"
    return redacted


def _line_key_indexes(lines: Iterable[str]) -> dict[str, int]:
    indexes: dict[str, int] = {}
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        indexes[key] = index
    return indexes


def _credential_value(credentials: Any, *names: str) -> str:
    if isinstance(credentials, dict):
        for name in names:
            if name in credentials:
                return str(credentials[name])
    for name in names:
        if hasattr(credentials, name):
            return str(getattr(credentials, name))
    raise LiveSetupError(f"Credential object did not include any of: {', '.join(names)}")


def _require_env(env: dict[str, str], keys: list[str]) -> None:
    missing = [key for key in keys if not env.get(key)]
    if missing:
        raise LiveSetupError(f"Missing required .env values: {', '.join(missing)}")


def _redact_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if len(value) <= 10:
        return "***"
    return f"{value[:6]}...{value[-4:]}"


def _redact_ip(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    parts = value.split(".")
    if len(parts) == 4:
        return ".".join(parts[:2] + ["x", "x"])
    return _redact_value(value)


def _env_mode(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return oct(path.stat().st_mode & 0o777)
