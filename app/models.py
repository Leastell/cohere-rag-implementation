# app/models.py - Models and types for typing

from dataclasses import dataclass
from pydantic import BaseModel
from pathlib import Path
from typing import TypedDict


@dataclass
class Document(BaseModel):
    id: str
    path: Path
    text: str


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    documents: list[dict]
