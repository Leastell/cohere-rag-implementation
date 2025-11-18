import pytest
from pathlib import Path
import numpy as np
import app.llm_utils as llm_utils


def test_load_prompt_invalid_dir(monkeypatch):
    ''' Verify that load_prompt throws error when prompt dir doesnt exist '''
    fake_dir = Path("not_a_real_dir")
    monkeypatch.setattr(llm_utils, "PROMPT_DIR", fake_dir)

    with pytest.raises(FileNotFoundError) as exc:
        llm_utils._load_prompt("any")

    assert str(fake_dir.resolve()) in str(exc.value)

def test_get_documents_invalid_dir(monkeypatch):
    ''' Verify that get_documents throws error when prompt dir doesnt exist '''
    fake_dir = Path("not_a_real_dir")
    monkeypatch.setattr(llm_utils, "DOCUMENT_DIR", fake_dir)

    with pytest.raises(FileNotFoundError) as exc:
        llm_utils._get_documents()

    assert str(fake_dir.resolve()) in str(exc.value)
    
def test_get_documents_returns_docs(monkeypatch, tmp_path):
    # Create a fake DOCUMENT_DIR with one text file
    file_content = "Hello, this is a test document."
    fake_file = tmp_path / "doc1.txt"
    fake_file.write_text(file_content, encoding="utf-8")
    monkeypatch.setattr(llm_utils, "DOCUMENT_DIR", tmp_path)

    docs = llm_utils._get_documents()
    assert isinstance(docs, list)
    assert len(docs) == 1
    doc = docs[0]
    assert doc["id"] == "doc1.txt"
    assert doc["path"] == fake_file
    assert doc["text"] == file_content

def test_create_embeddings_returns_embeddings(monkeypatch):
    # fake docs
    docs = [
        {"text": "refund policy text"},
        {"text": "shipping information text"}
    ]

    class FakeEmbeddings:
        def __init__(self, data):
            self.float = data

    class FakeResponse:
        def __init__(self, data):
            self.embeddings = FakeEmbeddings(data)

    fake_vectors = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    def fake_embed(texts, model, input_type, embedding_types):
        # Assert we called embed with the right inputs
        assert texts == [d["text"] for d in docs]
        assert input_type == "search_document"
        assert "float" in embedding_types
        # model should be whatever llm_utils.EMBED_MODEL is
        assert model == llm_utils.EMBED_MODEL
        return FakeResponse(fake_vectors)

    # Patch the co.embed method in llm_utils
    monkeypatch.setattr(llm_utils.co, "embed", fake_embed)

    # Call the function under test
    result = llm_utils._create_embeddings(docs)

    # Check result shape and values
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3)
    assert np.allclose(result, np.array(fake_vectors))


def test_load_prompt_success(monkeypatch, tmp_path):
    ''' Test that load_prompt successfully reads and returns file content '''
    prompt_content = "Context: {context}\nQuestion: {query}\nAnswer:"
    prompt_file = tmp_path / "response_prompt.txt"
    prompt_file.write_text(prompt_content, encoding="utf-8")
    monkeypatch.setattr(llm_utils, "PROMPT_DIR", tmp_path)

    result = llm_utils._load_prompt("response_prompt")
    assert result == prompt_content


def test_get_documents_skips_dirs(monkeypatch, tmp_path):
    ''' Test that get_documents ignores subdirectories and only processes files '''
    # Create a file and a subdirectory
    file_content = "Document content"
    fake_file = tmp_path / "doc.txt"
    fake_file.write_text(file_content, encoding="utf-8")
    fake_subdir = tmp_path / "subdir"
    fake_subdir.mkdir()
    
    monkeypatch.setattr(llm_utils, "DOCUMENT_DIR", tmp_path)

    docs = llm_utils._get_documents()
    assert len(docs) == 1  # Only the file, not the directory
    assert docs[0]["id"] == "doc.txt"


def test_create_embeddings_no_embeddings(monkeypatch):
    ''' Test create_embeddings raises error when API returns no embeddings '''
    docs = [{"text": "test"}]

    class FakeEmbeddings:
        def __init__(self):
            self.float = None

    class FakeResponse:
        def __init__(self):
            self.embeddings = FakeEmbeddings()

    def fake_embed(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(llm_utils.co, "embed", fake_embed)

    with pytest.raises(ValueError, match="Error fetching embeddings"):
        llm_utils._create_embeddings(docs)


def test_embedding_index_init():
    ''' Test EmbeddingIndex class initializes correctly with docs and embeddings '''
    docs = [{"id": "doc1", "text": "test"}]
    embeddings = np.array([[0.1, 0.2, 0.3]])
    
    index = llm_utils.EmbeddingIndex(docs, embeddings)
    assert index.docs == docs
    assert np.array_equal(index.embeddings, embeddings)


def test_build_embedding_index(monkeypatch):
    ''' Test complete build_embedding_index workflow from documents to index '''
    # Mock documents
    fake_docs = [{"text": "doc1"}, {"text": "doc2"}]
    fake_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])

    # Mock the internal functions
    monkeypatch.setattr(llm_utils, "_get_documents", lambda: fake_docs)
    monkeypatch.setattr(llm_utils, "_create_embeddings", lambda docs: fake_embeddings)

    result = llm_utils.build_embedding_index()
    
    assert isinstance(result, llm_utils.EmbeddingIndex)
    assert result.docs == fake_docs
    assert np.array_equal(result.embeddings, fake_embeddings)


def test_rank_documents():
    ''' Test document ranking by cosine similarity with correct ordering '''
    # Create test data
    docs = [
        {"id": "doc1", "text": "first document"},
        {"id": "doc2", "text": "second document"}
    ]
    embeddings = np.array([[0.1, 0.2], [0.8, 0.9]])
    index = llm_utils.EmbeddingIndex(docs, embeddings)
    
    # Query vector closer to second document
    query_vec = np.array([0.7, 0.8])
    
    results = llm_utils._rank_documents(query_vec, index, top_k=2)
    
    assert len(results) == 2
    # Check structure
    for result in results:
        assert "rank" in result
        assert "score" in result
        assert "id" in result
        assert "text" in result
    
    # Check ranking order (higher similarity first)
    assert results[0]["rank"] == 1
    assert results[1]["rank"] == 2
    assert results[0]["score"] >= results[1]["score"]
    # Second doc should be ranked higher (more similar)
    assert results[0]["id"] == "doc2"


def test_rank_documents_top_k():
    ''' Test that rank_documents respects top_k parameter and limits results '''
    docs = [{"id": f"doc{i}", "text": f"doc {i}"} for i in range(5)]
    embeddings = np.random.rand(5, 3)
    index = llm_utils.EmbeddingIndex(docs, embeddings)
    query_vec = np.array([0.5, 0.5, 0.5])
    
    results = llm_utils._rank_documents(query_vec, index, top_k=3)
    
    assert len(results) == 3  # Should limit to top_k
    assert all(result["rank"] <= 3 for result in results)


def test_answer_with_citations(monkeypatch):
    ''' Test complete answer_with_citations workflow with mocked API calls '''
    # Mock embedding index
    docs = [{"id": "doc1", "text": "Test document content"}]
    embeddings = np.array([[0.1, 0.2, 0.3]])
    mock_index = llm_utils.EmbeddingIndex(docs, embeddings)
    
    # Mock prompt
    mock_prompt = "Context: {context_block}\nQuery: {query}\nAnswer:"
    monkeypatch.setattr(llm_utils, "_load_prompt", lambda name: mock_prompt)
    
    # Mock query embedding
    class FakeEmbeddings:
        def __init__(self):
            self.float = [[0.1, 0.2, 0.3]]
    
    class FakeEmbedResponse:
        def __init__(self):
            self.embeddings = FakeEmbeddings()
    
    # Mock chat response
    class FakeContent:
        def __init__(self, text):
            self.text = text
    
    class FakeMessage:
        def __init__(self, content_text):
            self.content = [FakeContent(content_text)]
    
    class FakeChatResponse:
        def __init__(self, response_text):
            self.message = FakeMessage(response_text)
    
    def fake_embed(*args, **kwargs):
        return FakeEmbedResponse()
    
    def fake_chat(*args, **kwargs):
        return FakeChatResponse("This is a test answer")
    
    monkeypatch.setattr(llm_utils.co, "embed", fake_embed)
    monkeypatch.setattr(llm_utils.co, "chat", fake_chat)
    
    result = llm_utils.answer_with_citations("test query", mock_index)
    
    assert "answer" in result
    assert "documents" in result
    assert result["answer"] == "This is a test answer"
    assert len(result["documents"]) == 1
    assert result["documents"][0]["id"] == "doc1"
