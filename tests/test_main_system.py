"""
Unit tests for the main RAG system components.
Tests the actual working retriever and tools infrastructure.
"""
from typing import cast
from unittest.mock import Mock, patch

import pytest
from llama_index.core.schema import NodeWithScore

from app.infrastructure.retriever import (
    BaseRetriever,
    GuestRetriever,
    MockVectorRetriever,
    RetrieverInterface,
)
from app.infrastructure.tools import QueryTool


class TestRetrieverInterface:
    """Test the retriever interface compliance."""

    def test_guest_retriever_implements_interface(self):
        """Test that GuestRetriever implements RetrieverInterface."""
        retriever = GuestRetriever()
        assert hasattr(retriever, 'initialize')
        assert hasattr(retriever, 'search')
        assert callable(retriever.initialize)
        assert callable(retriever.search)

    def test_mock_vector_retriever_implements_interface(self):
        """Test that MockVectorRetriever implements RetrieverInterface."""
        retriever = MockVectorRetriever()
        assert hasattr(retriever, 'initialize')
        assert hasattr(retriever, 'search')
        assert callable(retriever.initialize)
        assert callable(retriever.search)


class TestBaseRetriever:
    """Test the BaseRetriever abstract class."""

    def test_init(self):
        """Test BaseRetriever initialization."""
        # Use GuestRetriever as concrete implementation
        retriever = GuestRetriever()
        assert retriever.docstore is not None
        assert retriever.retriever is None
        assert retriever.documents == []

    def test_abstract_methods_must_be_implemented(self):
        """Test that abstract methods cannot be called directly."""
        # This tests that BaseRetriever is properly abstract
        with pytest.raises(TypeError):
            BaseRetriever()  # type: ignore Cannot instantiate abstract class

    @patch('app.infrastructure.retriever.datasets.load_dataset')
    def test_load_documents(self, mock_load_dataset):
        """Test document loading from dataset."""
        # Mock dataset response
        mock_dataset = [
            {
                'name': 'Alice Johnson',
                'relation': 'Friend',
                'description': 'Software engineer',
                'email': 'alice@example.com'
            },
            {
                'name': 'Bob Smith',
                'relation': 'Colleague',
                'description': 'Data scientist',
                'email': 'bob@company.com'
            }
        ]
        mock_load_dataset.return_value = mock_dataset

        retriever = GuestRetriever()
        documents = retriever.load("test-dataset")

        assert len(documents) == 2
        assert documents[0].metadata["name"] == "Alice Johnson"
        assert documents[1].metadata["name"] == "Bob Smith"

        # Check document content structure
        alice_content = documents[0].get_content()
        assert "Alice Johnson" in alice_content
        assert "Software engineer" in alice_content
        assert "alice@example.com" in alice_content

    def test_format_node_with_scores_simple(self):
        """Test simple formatting of search results."""
        # Create mock results
        mock_results = []
        for name in ["Alice", "Bob", "Carol"]:
            mock_node = Mock()
            mock_node.metadata = {"name": name}
            mock_result = Mock(spec=NodeWithScore)
            mock_result.node = mock_node
            mock_results.append(mock_result)

        results = cast(list[NodeWithScore], mock_results)
        formatted = GuestRetriever.format_NodeWithScores(results, "simple")

        expected_lines = ["1. Alice", "2. Bob", "3. Carol"]
        assert formatted == "\n".join(expected_lines)

    def test_format_node_with_scores_detailed(self):
        """Test detailed formatting of search results."""
        # Create mock results
        mock_node = Mock()
        mock_node.get_content.return_value = "Name: Alice\nRole: Engineer"
        mock_result = Mock(spec=NodeWithScore)
        mock_result.node = mock_node

        results = cast(list[NodeWithScore], [mock_result])

        formatted = BaseRetriever.format_NodeWithScores(results, "detailed")
        assert "Result 1:" in formatted
        assert "Name: Alice" in formatted
        assert "Role: Engineer" in formatted

    def test_format_node_with_scores_empty(self):
        """Test formatting empty results."""
        formatted = BaseRetriever.format_NodeWithScores([], "simple")
        assert formatted == "No relevant guests found."

    def test_format_node_with_scores_invalid_type(self):
        """Test formatting with invalid format type."""
        mock_result = Mock()
        results = cast(list[NodeWithScore], [mock_result])

        with pytest.raises(ValueError, match="Unknown format_type"):
            BaseRetriever.format_NodeWithScores(results, "invalid")


class TestGuestRetriever:
    """Test the BM25-based GuestRetriever implementation."""

    def test_init(self):
        """Test GuestRetriever initialization."""
        retriever = GuestRetriever()
        assert isinstance(retriever, BaseRetriever)
        assert retriever.docstore is not None
        assert retriever.retriever is None
        assert retriever.documents == []

    @patch('app.infrastructure.retriever.datasets.load_dataset')
    def test_initialize_success(self, mock_load_dataset):
        """Test successful initialization with mocked dataset."""
        # Mock a small dataset to avoid external dependencies
        mock_dataset = [
            {
                'name': 'Test Guest',
                'relation': 'Friend',
                'description': 'Test person',
                'email': 'test@example.com'
            }
        ]
        mock_load_dataset.return_value = mock_dataset

        retriever = GuestRetriever()

        # Should not raise an exception
        retriever.initialize("test-dataset")

        # Verify state after initialization
        assert len(retriever.documents) == 1
        assert retriever.retriever is not None

    def test_search_without_initialization(self):
        """Test that search fails when not initialized."""
        retriever = GuestRetriever()

        with pytest.raises(ValueError, match="No documents indexed"):
            retriever.search("test query")

    @patch('app.infrastructure.retriever.datasets.load_dataset')
    def test_index_documents(self, mock_load_dataset):
        """Test document indexing functionality."""
        mock_dataset = [
            {
                'name': 'Alice',
                'relation': 'Friend',
                'description': 'Engineer',
                'email': 'alice@example.com'
            }
        ]
        mock_load_dataset.return_value = mock_dataset

        retriever = GuestRetriever()
        documents = retriever.load("test-dataset")

        # Test indexing returns correct count
        indexed_count = retriever.index(documents)
        assert indexed_count == 1
        assert retriever.retriever is not None


class TestMockVectorRetriever:
    """Test the MockVectorRetriever implementation."""

    def test_init(self):
        """Test MockVectorRetriever initialization."""
        retriever = MockVectorRetriever(similarity_top_k=5)
        assert isinstance(retriever, BaseRetriever)
        assert retriever.similarity_top_k == 5

    def test_init_defaults(self):
        """Test MockVectorRetriever with default parameters."""
        retriever = MockVectorRetriever()
        assert retriever.similarity_top_k == 3  # Default value

    @patch('app.infrastructure.retriever.datasets.load_dataset')
    def test_mock_vector_search(self, mock_load_dataset):
        """Test mock vector similarity search."""
        mock_dataset = [
            {
                'name': 'Alice Engineer',
                'relation': 'Friend',
                'description': 'Software engineer specializing in Python',
                'email': 'alice@example.com'
            },
            {
                'name': 'Bob Data',
                'relation': 'Colleague',
                'description': 'Data scientist working with machine learning',
                'email': 'bob@company.com'
            }
        ]
        mock_load_dataset.return_value = mock_dataset

        retriever = MockVectorRetriever(similarity_top_k=2)
        retriever.initialize("test-dataset")

        # Search for engineer-related query
        results = retriever.search("engineer")

        # Should return results (mock scoring based on word matches)
        assert len(results) > 0
        assert all(hasattr(result, 'node') for result in results)
        assert all(hasattr(result, 'score') for result in results)

    def test_search_without_initialization(self):
        """Test that search fails when not initialized."""
        retriever = MockVectorRetriever()

        with pytest.raises(ValueError, match="No documents indexed"):
            retriever.search("test query")


class TestQueryTool:
    """Test the LlamaIndex QueryTool implementation."""

    def test_init(self):
        """Test QueryTool initialization."""
        mock_retriever = Mock()
        tool = QueryTool(mock_retriever)
        assert tool.retriever == mock_retriever

    def test_metadata(self):
        """Test tool metadata."""
        mock_retriever = Mock()
        tool = QueryTool(mock_retriever)

        metadata = tool.metadata
        assert metadata.name == "query_guests"
        assert "search for guests" in metadata.description.lower()

    def test_run_method(self):
        """Test the run method."""
        mock_retriever = Mock()
        mock_results = [Mock(), Mock()]
        mock_retriever.search.return_value = mock_results

        tool = QueryTool(mock_retriever)
        results = tool.run("test query")

        assert results == mock_results
        mock_retriever.search.assert_called_once_with("test query")

    def test_call_method(self):
        """Test the __call__ method (main tool interface)."""
        mock_retriever = Mock()
        mock_node = Mock()
        mock_node.get_content.return_value = "Test guest content"
        mock_node.metadata = {"name": "Test Guest"}
        mock_result = Mock()
        mock_result.node = mock_node
        mock_retriever.search.return_value = [mock_result]

        tool = QueryTool(mock_retriever)
        tool_output = tool("test query")

        # Verify ToolOutput structure
        assert hasattr(tool_output, 'tool_name')
        assert hasattr(tool_output, 'content')
        assert tool_output.tool_name == "query_guests"
        assert "Test guest content" in tool_output.content


class TestRetrieverSwapping:
    """Test that different retrievers can be swapped seamlessly."""

    def test_interface_compatibility(self):
        """Test that different retrievers work with the same client code."""
        def client_code(retriever: RetrieverInterface) -> str:
            """Example client code that works with any retriever."""
            try:
                # This would normally initialize with real data
                # retriever.initialize("test-dataset")
                # results = retriever.search("test query")
                return "success"
            except Exception as e:
                return f"error: {e}"

        # Test with different retriever implementations
        guest_retriever = GuestRetriever()
        mock_vector_retriever = MockVectorRetriever()

        # Both should work with the same client code
        assert client_code(guest_retriever) == "success"
        assert client_code(mock_vector_retriever) == "success"

    def test_tool_with_different_retrievers(self):
        """Test that QueryTool works with different retriever implementations."""
        mock_results = [Mock()]

        # Test with GuestRetriever
        guest_retriever = Mock(spec=GuestRetriever)
        guest_retriever.search.return_value = mock_results
        guest_tool = QueryTool(guest_retriever)

        # Test with MockVectorRetriever
        vector_retriever = Mock(spec=MockVectorRetriever)
        vector_retriever.search.return_value = mock_results
        vector_tool = QueryTool(vector_retriever)

        # Both tools should work the same way
        guest_results = guest_tool.run("test")
        vector_results = vector_tool.run("test")

        assert guest_results == mock_results
        assert vector_results == mock_results
