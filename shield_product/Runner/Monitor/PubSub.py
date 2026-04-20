from typing import Dict, List


class PubSub:
    def __init__(self) -> None:
        self._items: List[Dict[str, object]] = []

    def publish(self, payload: Dict[str, object]) -> None:
        self._items.append(payload)

    def all(self) -> List[Dict[str, object]]:
        return list(self._items)
