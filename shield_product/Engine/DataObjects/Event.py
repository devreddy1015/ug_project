from dataclasses import dataclass, field
from typing import List

from .Base import BaseDataObject


@dataclass
class Event:
    items: List[BaseDataObject] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
