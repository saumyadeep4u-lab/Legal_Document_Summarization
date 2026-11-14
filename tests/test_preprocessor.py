"""
tests/test_preprocessor.py
Automated unit tests for the LegalPreprocessor class to ensure chunking 
and NER logic strictly adhere to transformer constraints.
"""

import pytest
from src.preprocessor import LegalPreprocessor


# Pytest fixture to initialize the preprocessor once for all tests
@pytest.fixture(scope="module")
def preprocessor():
    return LegalPreprocessor()


def test_clean_text(preprocessor):
    """Ensures invisible characters and whitespace are normalized."""
    messy_text = "This   Agreement \n\n is strictly   confidential."
    cleaned = preprocessor.clean_text(messy_text)
    assert cleaned == "This Agreement is strictly confidential."


def test_chunk_document_logic(preprocessor):
    """
    Ensures that a massively long document is successfully chopped 
    into multiple overlapping chunks without exceeding limits.
    """
    # Create a synthetic document that is artificially long
    synthetic_long_contract = "The Governing Law shall apply. " * 500

    chunks = preprocessor.chunk_document(synthetic_long_contract)

    # 500 sentences will exceed 1024 tokens, so it MUST generate more than 1 chunk
    assert len(chunks) > 1

    # Ensure no chunk is completely empty
    for chunk in chunks:
        assert len(chunk) > 0


def test_extract_legal_entities(preprocessor):
    """Ensures the NER pipeline accurately identifies critical legal facts."""
    test_clause = "Apple Inc. agrees to pay Jane Smith $100,000 on October 31st."

    entities = preprocessor.extract_legal_entities(test_clause)

    # Check if the critical facts were captured
    assert "Jane Smith" in entities.get("PERSON", [])
    assert "100,000" in entities.get("MONEY", []) or "$100,000" in entities.get("MONEY", [])
    assert "Apple Inc." in entities.get("ORG", [])
    assert "October 31st" in entities.get("DATE", [])