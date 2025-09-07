from __future__ import annotations

import json as _json
import logging
import random
import time
from typing import Any, Dict, Optional, Tuple
from urllib import error, request

from .errors import ProviderError


log = logging.getLogger("http")


def _join_url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def _should_retry(
    status: Optional[int],
    exc: Optional[BaseException],
    *,
    enabled: bool,
    status_codes: Optional[set[int]],
    include_5xx: bool,
) -> bool:
    if not enabled:
        return False
    if status is not None:
        if status_codes and status in status_codes:
            return True
        if include_5xx and 500 <= status < 600:
            return True
    if exc is not None:
        # URLError, timeouts, transient network failures
        return True
    return False


def _sleep_backoff(attempt: int, *, base_ms: int, factor: float, jitter: bool) -> None:
    delay = base_ms * (factor ** max(attempt - 1, 0)) / 1000.0
    if jitter:
        delay *= (0.5 + random.random())  # 0.5x to 1.5x
    time.sleep(min(delay, 10))  # cap to 10s between tries


def post_json(
    *,
    base_url: str,
    path: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout_s: float,
    max_attempts: int,
    backoff: Tuple[int, float, bool],
    accept_sse: bool = False,
    retry_enabled: bool = True,
    retry_status_codes: Optional[set[int]] = None,
    retry_include_5xx: bool = True,
) -> request.addinfourl:
    url = _join_url(base_url, path)
    data = _json.dumps(payload).encode("utf-8")
    if accept_sse:
        headers = {**headers, "Accept": "text/event-stream"}
    req = request.Request(url, data=data, headers=headers, method="POST")

    base_ms, factor, jitter = backoff
    attempt = 0
    last_status: Optional[int] = None
    last_exc: Optional[BaseException] = None
    while attempt < max_attempts:
        attempt += 1
        try:
            log.debug("POST %s attempt=%s", url, attempt)
            resp = request.urlopen(req, timeout=timeout_s)
            return resp
        except error.HTTPError as e:
            last_status = e.code
            body = e.read().decode("utf-8", errors="ignore")
            log.warning("HTTPError %s on %s: %s", e.code, url, body)
            if attempt >= max_attempts or not _should_retry(e.code, None, enabled=retry_enabled, status_codes=retry_status_codes, include_5xx=retry_include_5xx):
                raise ProviderError("HTTPError", status=e.code, body=body)
        except error.URLError as e:
            last_exc = e
            log.warning("URLError on %s: %s", url, e)
            if attempt >= max_attempts or not _should_retry(None, e, enabled=retry_enabled, status_codes=retry_status_codes, include_5xx=retry_include_5xx):
                raise ProviderError(f"Network error: {e}")
        _sleep_backoff(attempt, base_ms=base_ms, factor=factor, jitter=jitter)

    # If we exit loop without return/raise (shouldn't happen), raise generic
    raise ProviderError("Request failed", status=last_status, body=str(last_exc) if last_exc else None)
