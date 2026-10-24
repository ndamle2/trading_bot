"""A small client for the Archipelago Exchange: REST calls, the market-data WebSocket, order-action rate limiting.

    from client import Exchange
    ex = Exchange()                    # key from $HURRICANE_API_KEY or trading_comp/.env
    ex.time(); ex.positions(); ex.orderbook("PORT")
    oid = ex.limit("PORT", "buy", 1, 10.0); ex.cancel(oid)
    async for msg in ex.stream(["news", "quotes"]): ...
"""

from __future__ import annotations

import asyncio
import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, AsyncIterator

import certifi
import websockets

# python.org builds on macOS ship without a CA bundle; use certifi's for HTTPS and WSS
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
DEFAULT_URL = "https://cornelltradingcompetition.org"
# the network in front of the exchange refuses the default urllib user agent
USER_AGENT = "trading-comp-bot/0.1"


def load_key() -> str:
    key = os.environ.get("HURRICANE_API_KEY")
    env = Path(__file__).resolve().parent.parent / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HURRICANE_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        raise SystemExit("no bot key: set HURRICANE_API_KEY or put HURRICANE_API_KEY=... in trading_comp/.env")
    return key


class RateLimiter:
    """At most `n` order actions in any rolling 60 s window (the team limit is 180/min)."""

    def __init__(self, n: int) -> None:
        self.n, self.stamps, self.lock = n, deque(), threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            while self.stamps and now - self.stamps[0] > 60:
                self.stamps.popleft()
            if len(self.stamps) >= self.n:
                time.sleep(60 - (now - self.stamps[0]) + 0.01)
            self.stamps.append(time.monotonic())


class ApiError(Exception):
    def __init__(self, code: int, body: str) -> None:
        super().__init__(f"HTTP {code}: {body[:300]}")
        self.code, self.body = code, body


class Exchange:
    def __init__(self, url: str = DEFAULT_URL, key: str | None = None, actions_per_min: int = 150) -> None:
        self.url = url.rstrip("/")
        self.key = key or load_key()
        self.limiter = RateLimiter(actions_per_min)

    # ---- REST
    def request(self, method: str, path: str, body: Any = None, api: bool = True) -> Any:
        headers = {"User-Agent": USER_AGENT, "X-API-Key": self.key}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.url + ("/api/v1" if api else "") + path, data=data, method=method,
                                     headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10, context=SSL_CONTEXT) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            raise ApiError(exc.code, exc.read().decode(errors="replace")) from None

    def case(self) -> dict:
        return self.request("GET", "/cases/hurricane.json", api=False)

    def time(self) -> dict:
        return self.request("GET", "/competitions/time")

    def positions(self) -> Any:
        return self.request("GET", "/positions")

    def open_orders(self) -> Any:
        return self.request("GET", "/orders/open")

    def orderbook(self, symbol: str) -> Any:
        return self.request("GET", f"/orderbook/{symbol}")

    # ---- order actions (rate limited)
    def place(self, symbol: str, side: str, quantity: int, price: float | None = None) -> dict:
        body = {"symbol": symbol, "side": side, "order_type": "limit" if price is not None else "market",
                "quantity": int(quantity)}
        if price is not None:
            body["price"] = round(float(price), 6)
        self.limiter.wait()
        return self.request("POST", "/orders", body)

    def limit(self, symbol: str, side: str, quantity: int, price: float) -> dict:
        return self.place(symbol, side, quantity, price)

    def cancel(self, order_id: Any) -> Any:
        self.limiter.wait()
        return self.request("DELETE", f"/orders/{order_id}")

    def cancel_all(self, symbol: str | None = None) -> Any:
        self.limiter.wait()
        return self.request("DELETE", "/orders/all" + (f"?symbol={symbol}" if symbol else ""))

    def create_etf(self, symbol: str, quantity: int = 1) -> Any:
        self.limiter.wait()
        return self.request("POST", f"/etf/{symbol}/create", {"quantity": quantity})

    def redeem_etf(self, symbol: str, quantity: int = 1) -> Any:
        self.limiter.wait()
        return self.request("POST", f"/etf/{symbol}/redeem", {"quantity": quantity})

    # ---- market data
    async def stream(self, channels: list[str], symbols: list[str] | None = None) -> AsyncIterator[dict]:
        """Yield every frame from /ws/v1/market-data forever, reconnecting with backoff."""
        ws_url = self.url.replace("https://", "wss://", 1).replace("http://", "ws://", 1) + "/ws/v1/market-data"
        sub = {"action": "subscribe", "symbols": symbols or [], "channels": channels}
        backoff = 1.0
        while True:
            try:
                ssl_ctx = SSL_CONTEXT if ws_url.startswith("wss://") else None
                async with websockets.connect(ws_url, ssl=ssl_ctx, user_agent_header=USER_AGENT,
                                              additional_headers={"X-API-Key": self.key}) as ws:
                    await ws.send(json.dumps(sub))
                    backoff = 1.0
                    async for frame in ws:
                        yield json.loads(frame)
            except (OSError, websockets.WebSocketException) as exc:
                print(f"[ws] disconnected ({exc!r}); retry in {backoff:.0f}s", flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


def order_id(resp: Any) -> Any:
    """The id of an order in a POST /orders response (field name not documented, so check the usual ones)."""
    if isinstance(resp, dict):
        for k in ("order_id", "id", "orderId"):
            if k in resp:
                return resp[k]
        for v in resp.values():
            if isinstance(v, dict):
                found = order_id(v)
                if found is not None:
                    return found
    return None
