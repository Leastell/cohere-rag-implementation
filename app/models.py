# app/models.py - Models and types for typing

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict


@dataclass
class Document:
    id: str
    path: Path
    text: str
