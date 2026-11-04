"""Журнал шагов одного запроса."""

from typing import Any


class Journal:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.attempts = 0

    def add(self, step: str, **payload: Any) -> None:
        event = {"step": step, **payload}
        self.events.append(event)

    def steps(self) -> list[str]:
        return [event["step"] for event in self.events]
