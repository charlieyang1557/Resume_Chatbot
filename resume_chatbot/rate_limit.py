"""In-memory rate limiting for the chat API.

Two limits:
- Per-IP: max 20 requests per hour (sliding window).
- Global daily: max 100 LLM calls per day, resets at midnight UTC.
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timezone
from threading import Lock


class RateLimiter:
    """Simple in-memory rate limiter with per-IP and global daily caps."""

    def __init__(
        self,
        per_ip_limit: int = 20,
        per_ip_window: int = 3600,
        daily_limit: int = 100,
    ) -> None:
        self._per_ip_limit = per_ip_limit
        self._per_ip_window = per_ip_window
        self._daily_limit = daily_limit

        self._ip_requests: dict[str, list[float]] = defaultdict(list)
        self._daily_count = 0
        self._daily_date: str = self._today()
        self._lock = Lock()

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _reset_daily_if_needed(self) -> None:
        today = self._today()
        if self._daily_date != today:
            self._daily_count = 0
            self._daily_date = today

    def check(self, ip: str) -> str | None:
        """Return an error message if the request should be rejected, else None."""
        with self._lock:
            now = time.time()
            self._reset_daily_if_needed()

            # Global daily cap
            if self._daily_count >= self._daily_limit:
                return "Daily limit reached, try again tomorrow."

            # Per-IP sliding window
            timestamps = self._ip_requests[ip]
            cutoff = now - self._per_ip_window
            timestamps[:] = [t for t in timestamps if t > cutoff]

            if len(timestamps) >= self._per_ip_limit:
                return "Rate limit exceeded. Please wait before sending more messages."

            # Allow request
            timestamps.append(now)
            self._daily_count += 1
            return None

    @property
    def daily_count(self) -> int:
        with self._lock:
            self._reset_daily_if_needed()
            return self._daily_count

    @property
    def daily_limit(self) -> int:
        return self._daily_limit
