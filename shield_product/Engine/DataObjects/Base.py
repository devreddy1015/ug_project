from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class BaseDataObject:
    path: Path
    modality: str
    label: Optional[int] = None
    metadata: Dict[str, object] = field(default_factory=dict)
