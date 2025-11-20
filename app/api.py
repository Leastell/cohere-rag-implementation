from fastapi import APIRouter, Depends, Request

router = APIRouter()


def get_rag_service(request: Request):
    return request.app.state.rag_service


@router.post("/query")
async def query_endpoint(body: dict, rag=Depends(get_rag_service)):
    query = body["query"]
    return rag.answer_with_citations(query)
