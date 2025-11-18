# app/main.py - TODO: define file scope

from .llm_utils import build_embedding_index
from .api import router
from contextlib import asynccontextmanager
from fastapi import FastAPI
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup operations:
    #   Build embedding index
    app.state.embedding_index = build_embedding_index()
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(router)
