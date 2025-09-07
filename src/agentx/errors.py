from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ProviderError(Exception):
    message: str
    status: Optional[int] = None
    body: Optional[str] = None

    def __str__(self) -> str:
        base = self.message
        if self.status is not None:
            base += f" (status {self.status})"
        return base

