import dataclasses
from typing import List, Union, Dict, Any, Mapping, Optional


@dataclasses
class Config:
    batch_size = field(default=2000)
    embedding_size

