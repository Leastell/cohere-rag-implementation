# app/api.py - API endpoints for app services

from fastapi import APIRouter, Depends, Request
from .llm_utils import answer_with_citations

router = APIRouter()

def get_embedding_index(request: Request):
    return request.app.state.embedding_index

@router.post("/query")
async def query_endpoint(
    body: dict,
    index = Depends(get_embedding_index),
):
    # answer_with_citations(query, index)
    query = body["query"]      # from your JSON body
    result = answer_with_citations(query, index)

    return result