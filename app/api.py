from fastapi import APIRouter, Depends, HTTPException, Request
from .models import QueryRequest, QueryResponse

router = APIRouter()


def get_rag_service(request: Request):
    return request.app.state.rag_service


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, rag=Depends(get_rag_service)):
    try:
        return rag.answer_with_citations(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
