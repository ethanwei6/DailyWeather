from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from weather_strategy.models import TradeSignal


class ExecutionAdapter:
    def place_order(self, signal: TradeSignal) -> str:
        raise NotImplementedError


class DisabledLiveExecutionAdapter(ExecutionAdapter):
    def place_order(self, signal: TradeSignal) -> str:
        raise RuntimeError(
            "Live execution is disabled. Paper trade this strategy first, then add a Polymarket API/MCP adapter here."
        )


@dataclass(frozen=True)
class LiveOrderResult:
    action: str
    token_id: str
    requested_notional_usd: float
    requested_shares: float
    filled_notional_usd: float
    filled_shares: float
    average_price: float
    order_id: Optional[str]
    status: str
    transaction_hashes: tuple[str, ...]
    raw_response: dict[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "execution_mode": "live",
            "order_id": self.order_id,
            "status": self.status,
            "transaction_hashes": list(self.transaction_hashes),
            "requested_notional_usd": self.requested_notional_usd,
            "requested_shares": self.requested_shares,
            "filled_notional_usd": self.filled_notional_usd,
            "filled_shares": self.filled_shares,
            "average_price": self.average_price,
            "raw_response": self.raw_response,
        }


class PolymarketLiveExecutionAdapter(ExecutionAdapter):
    """Guarded Polymarket CLOB market-order adapter for live bankroll tests."""

    def __init__(self, env_file: str = ".env.local", *, min_collateral_reserve_usd: float = 0.0):
        self.env_file = Path(env_file)
        self.env = _read_env_file(self.env_file)
        if self.env.get("DAILYWEATHER_LIVE_TRADING") != "1":
            raise RuntimeError("DAILYWEATHER_LIVE_TRADING must be 1 before live orders are allowed")
        self.max_bankroll_usd = _env_float(self.env, "DAILYWEATHER_MAX_BANKROLL_USD", 50.0)
        self.max_order_usd = _env_float(self.env, "DAILYWEATHER_MAX_ORDER_USD", 5.0)
        self.min_collateral_reserve_usd = max(0.0, float(min_collateral_reserve_usd))
        if self.max_bankroll_usd <= 0 or self.max_order_usd <= 0:
            raise RuntimeError("Live bankroll and order caps must be positive")
        self._client = None
        self._allowance_retry_attempted = False

    def place_order(self, signal: TradeSignal) -> str:
        raise RuntimeError("Use execute_rebalance_order for Kelly live execution")

    def execute_rebalance_order(self, order: dict[str, Any]) -> LiveOrderResult:
        action = str(order["action"]).upper()
        token_id = str(order["token_id"])
        requested_notional = float(order.get("notional_usd") or 0.0)
        requested_shares = float(order.get("shares") or 0.0)
        if action not in {"BUY", "SELL"}:
            raise RuntimeError(f"Unsupported live action: {action}")
        if requested_notional <= 0 or requested_shares <= 0:
            raise RuntimeError(f"Invalid live order sizing: {order!r}")

        client = self._clob_client()
        balance_usd = self.collateral_balance_usd()
        cap_metadata = None
        requested_notional, requested_shares, cap_metadata = _cap_live_buy_order(
            action,
            requested_notional,
            requested_shares,
            max_order_usd=self.max_order_usd,
            max_bankroll_usd=self.max_bankroll_usd,
            balance_usd=balance_usd,
            min_collateral_reserve_usd=self.min_collateral_reserve_usd,
        )

        try:
            from py_clob_client_v2.clob_types import MarketOrderArgsV2
            from py_clob_client_v2.constants import BYTES32_ZERO
            from py_clob_client_v2.clob_types import OrderType
        except ImportError as error:  # pragma: no cover - live dependency guard
            raise RuntimeError("py-clob-client-v2 is required for live execution") from error

        builder_code = self.env.get("POLYMARKET_BUILDER_CODE") or BYTES32_ZERO
        amount = requested_notional if action == "BUY" else requested_shares
        kwargs: dict[str, Any] = {
            "token_id": token_id,
            "amount": amount,
            "side": action,
            "order_type": OrderType.FOK,
            "builder_code": builder_code,
        }
        if action == "BUY":
            kwargs["user_usdc_balance"] = balance_usd
        try:
            response = client.create_and_post_market_order(MarketOrderArgsV2(**kwargs), order_type=OrderType.FOK)
        except Exception as error:
            if not self._should_retry_after_allowance_error(error):
                raise
            self._allowance_retry_attempted = True
            self.refresh_deposit_wallet_allowances()
            response = client.create_and_post_market_order(MarketOrderArgsV2(**kwargs), order_type=OrderType.FOK)
        response_for_parse = dict(response)
        if cap_metadata is not None:
            response_for_parse["dailyweather_order_cap"] = cap_metadata
        return _parse_live_order_response(action, token_id, requested_notional, requested_shares, response_for_parse)

    def collateral_balance_usd(self) -> float:
        try:
            from py_clob_client_v2.clob_types import AssetType, BalanceAllowanceParams
        except ImportError as error:  # pragma: no cover - live dependency guard
            raise RuntimeError("py-clob-client-v2 is required for live execution") from error
        payload = self._clob_client().get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        return float(payload.get("balance") or 0) / 1_000_000

    def _clob_client(self):
        if self._client is not None:
            return self._client
        required = (
            "PRIVATE_KEY",
            "CLOB_API_KEY",
            "CLOB_SECRET",
            "CLOB_PASS_PHRASE",
            "DEPOSIT_WALLET_ADDRESS",
            "POLYMARKET_SIGNATURE_TYPE",
        )
        missing = [key for key in required if not self.env.get(key)]
        if missing:
            raise RuntimeError(f"Missing live .env values: {', '.join(missing)}")
        try:
            from py_clob_client_v2 import ApiCreds, ClobClient
            from py_clob_client_v2.clob_types import BuilderConfig
            from py_clob_client_v2.constants import POLYGON
        except ImportError as error:  # pragma: no cover - live dependency guard
            raise RuntimeError("py-clob-client-v2 is required for live execution") from error

        creds = ApiCreds(
            api_key=self.env["CLOB_API_KEY"],
            api_secret=self.env["CLOB_SECRET"],
            api_passphrase=self.env["CLOB_PASS_PHRASE"],
        )
        builder_config = BuilderConfig(
            builder_address=self.env.get("POLYMARKET_BUILDER_ADDRESS") or "",
            builder_code=self.env.get("POLYMARKET_BUILDER_CODE") or "",
        )
        self._client = ClobClient(
            self.env.get("CLOB_API_URL") or "https://clob.polymarket.com",
            chain_id=int(self.env.get("CHAIN_ID") or POLYGON),
            key=self.env["PRIVATE_KEY"],
            creds=creds,
            signature_type=int(self.env["POLYMARKET_SIGNATURE_TYPE"]),
            funder=self.env["DEPOSIT_WALLET_ADDRESS"],
            builder_config=builder_config,
            retry_on_error=True,
        )
        return self._client

    def refresh_deposit_wallet_allowances(self) -> dict[str, Any]:
        if self.env.get("DAILYWEATHER_AUTO_APPROVE_ALLOWANCES") != "1":
            raise RuntimeError(
                "Live order failed because allowance is insufficient and DAILYWEATHER_AUTO_APPROVE_ALLOWANCES is not 1"
            )
        required = (
            "PRIVATE_KEY",
            "BUILDER_API_KEY",
            "BUILDER_SECRET",
            "BUILDER_PASS_PHRASE",
            "RELAYER_URL",
            "CHAIN_ID",
            "DEPOSIT_WALLET_ADDRESS",
        )
        missing = [key for key in required if not self.env.get(key)]
        if missing:
            raise RuntimeError(f"Missing live allowance .env values: {', '.join(missing)}")
        try:
            from py_builder_relayer_client.client import RelayClient
            from py_builder_relayer_client.models import DepositWalletCall, RelayerTransactionState, TransactionType
            from py_builder_signing_sdk.config import BuilderConfig as RelayerBuilderConfig
            from py_builder_signing_sdk.sdk_types import BuilderApiKeyCreds
            from py_clob_client_v2.config import get_contract_config
        except ImportError as error:  # pragma: no cover - live dependency guard
            raise RuntimeError("py-builder-relayer-client and py-clob-client-v2 are required for live approvals") from error

        chain_id = int(self.env.get("CHAIN_ID") or 137)
        clob_config = get_contract_config(chain_id)
        builder_config = RelayerBuilderConfig(
            local_builder_creds=BuilderApiKeyCreds(
                key=self.env["BUILDER_API_KEY"],
                secret=self.env["BUILDER_SECRET"],
                passphrase=self.env["BUILDER_PASS_PHRASE"],
            )
        )
        client = RelayClient(
            relayer_url=self.env["RELAYER_URL"],
            chain_id=chain_id,
            private_key=self.env["PRIVATE_KEY"],
            builder_config=builder_config,
        )
        deposit_wallet = self.env["DEPOSIT_WALLET_ADDRESS"]
        expected_wallet = client.get_expected_deposit_wallet()
        if deposit_wallet.lower() != expected_wallet.lower():
            raise RuntimeError(
                f"DEPOSIT_WALLET_ADDRESS {deposit_wallet} does not match signer-derived wallet {expected_wallet}"
            )
        if not client.get_deployed(deposit_wallet, TransactionType.WALLET.value):
            raise RuntimeError(f"Deposit wallet is not deployed: {deposit_wallet}")

        signer_address = client.signer.address()
        nonce_payload = client.get_nonce(signer_address, TransactionType.WALLET.value)
        nonce = nonce_payload.get("nonce") if isinstance(nonce_payload, dict) else None
        if nonce is None:
            raise RuntimeError(f"Invalid relayer nonce payload: {nonce_payload!r}")

        calls = _dedupe_calls(
            [
                DepositWalletCall(
                    target=clob_config.collateral,
                    value="0",
                    data=_encode_erc20_approve(clob_config.exchange_v2, _max_uint256()),
                ),
                DepositWalletCall(
                    target=clob_config.collateral,
                    value="0",
                    data=_encode_erc20_approve(clob_config.neg_risk_exchange_v2, _max_uint256()),
                ),
                DepositWalletCall(
                    target=clob_config.conditional_tokens,
                    value="0",
                    data=_encode_set_approval_for_all(clob_config.exchange_v2, True),
                ),
                DepositWalletCall(
                    target=clob_config.conditional_tokens,
                    value="0",
                    data=_encode_set_approval_for_all(clob_config.neg_risk_exchange_v2, True),
                ),
            ]
        )
        deadline_seconds = int(_env_float(self.env, "DAILYWEATHER_ALLOWANCE_DEADLINE_SECONDS", 3600.0))
        response = client.execute_deposit_wallet_batch(
            calls=calls,
            wallet_address=deposit_wallet,
            nonce=str(nonce),
            deadline=str(int(time.time()) + max(300, deadline_seconds)),
        )
        poll_result = client.poll_until_state(
            transaction_id=response.transaction_id,
            states=[RelayerTransactionState.STATE_MINED.value, RelayerTransactionState.STATE_CONFIRMED.value],
            fail_state=RelayerTransactionState.STATE_FAILED.value,
            max_polls=30,
            poll_frequency=1000,
        )
        if not isinstance(poll_result, dict):
            raise RuntimeError(f"Allowance refresh did not reach mined/confirmed state: {response.transaction_id}")
        return {
            "transaction_id": response.transaction_id,
            "transaction_hash": response.transaction_hash,
            "state": poll_result.get("state"),
            "calls": len(calls),
        }

    def _should_retry_after_allowance_error(self, error: Exception) -> bool:
        if self._allowance_retry_attempted:
            return False
        if self.env.get("DAILYWEATHER_AUTO_APPROVE_ALLOWANCES") != "1":
            return False
        message = str(error).lower()
        return "allowance" in message and "not enough balance" in message


def _cap_live_buy_order(
    action: str,
    requested_notional: float,
    requested_shares: float,
    *,
    max_order_usd: float,
    max_bankroll_usd: float,
    balance_usd: float,
    min_collateral_reserve_usd: float = 0.0,
) -> tuple[float, float, Optional[dict[str, Any]]]:
    if action != "BUY":
        return requested_notional, requested_shares, None
    available_balance_usd = max(0.0, balance_usd - max(0.0, min_collateral_reserve_usd))
    capped_notional = min(requested_notional, max_order_usd, max_bankroll_usd, available_balance_usd)
    if capped_notional <= 0:
        raise RuntimeError(
            f"Live BUY ${requested_notional:.2f} exceeds available capped collateral "
            f"${available_balance_usd:.2f} after ${min_collateral_reserve_usd:.2f} reserve"
        )
    if capped_notional >= requested_notional - 1e-9:
        return requested_notional, requested_shares, None
    scale = capped_notional / requested_notional
    capped_shares = requested_shares * scale
    return (
        capped_notional,
        capped_shares,
        {
            "downsized_to_safety_cap": True,
            "original_requested_notional_usd": requested_notional,
            "original_requested_shares": requested_shares,
            "capped_requested_notional_usd": capped_notional,
            "capped_requested_shares": capped_shares,
            "max_order_usd": max_order_usd,
            "max_bankroll_usd": max_bankroll_usd,
            "collateral_balance_usd": balance_usd,
            "min_collateral_reserve_usd": min_collateral_reserve_usd,
            "available_after_reserve_usd": available_balance_usd,
        },
    )


def _parse_live_order_response(
    action: str,
    token_id: str,
    requested_notional: float,
    requested_shares: float,
    response: dict[str, Any],
) -> LiveOrderResult:
    if not isinstance(response, dict):
        raise RuntimeError(f"Unexpected live order response: {response!r}")
    if not response.get("success"):
        raise RuntimeError(f"Live order failed: {response!r}")
    making = float(response.get("makingAmount") or 0)
    taking = float(response.get("takingAmount") or 0)
    if action == "BUY":
        filled_notional = making
        filled_shares = taking
    else:
        filled_shares = making
        filled_notional = taking
    if filled_notional <= 0 or filled_shares <= 0:
        raise RuntimeError(f"Live order returned zero fill: {response!r}")
    return LiveOrderResult(
        action=action,
        token_id=token_id,
        requested_notional_usd=requested_notional,
        requested_shares=requested_shares,
        filled_notional_usd=filled_notional,
        filled_shares=filled_shares,
        average_price=filled_notional / filled_shares,
        order_id=response.get("orderID"),
        status=str(response.get("status") or "unknown"),
        transaction_hashes=tuple(str(item) for item in (response.get("transactionsHashes") or [])),
        raw_response=response,
    )


def _max_uint256() -> int:
    return (1 << 256) - 1


def _encode_erc20_approve(spender: str, amount: int) -> str:
    return "0x095ea7b3" + _encode_address(spender) + _encode_uint256(amount)


def _encode_set_approval_for_all(operator: str, approved: bool) -> str:
    return "0xa22cb465" + _encode_address(operator) + _encode_uint256(1 if approved else 0)


def _encode_address(address: str) -> str:
    normalized = address.lower().removeprefix("0x")
    if len(normalized) != 40:
        raise RuntimeError(f"Invalid address for calldata encoding: {address}")
    return normalized.rjust(64, "0")


def _encode_uint256(value: int) -> str:
    if value < 0:
        raise RuntimeError("uint256 value must be non-negative")
    return hex(value)[2:].rjust(64, "0")


def _dedupe_calls(calls: list[Any]) -> list[Any]:
    deduped = []
    seen = set()
    for call in calls:
        key = (call.target.lower(), str(call.value), call.data.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(call)
    return deduped


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        raise RuntimeError(f"Live env file does not exist: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _env_float(env: dict[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key) or default)
    except ValueError as error:
        raise RuntimeError(f"{key} must be numeric") from error
