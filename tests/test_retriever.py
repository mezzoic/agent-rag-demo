"""
Unit tests for the retriever infrastructure.
Tests the core retrieval functionality without requiring external APIs.
"""
from unittest.mock import Mock, patch

import pytest
from llama_index.core.schema import Document, NodeWithScore
from llama_index.core.storage.docstore import SimpleDocumentStore

from app.infrastructure.retriever import (
    BaseRetriever,
    GuestRetriever,
    MockVectorRetriever,
    RetrieverInterface,
)


class TestRetrieverInterface:
    """Test the Protocol interface works correctly."""

    def test_interface_methods_exist(self):
        """Test that RetrieverInterface has required methods."""
        # This is a Protocol, so we can't instantiate it directly
        # But we can check that our implementations satisfy it
        retriever = GuestRetriever()
        assert hasattr(retriever, 'initialize')
        assert hasattr(retriever, 'search')
        assert callable(retriever.initialize)
        assert callable(retriever.search)


class TestBaseRetriever:
    """Test the abstract base class functionality."""

    def test_init(self):
        """Test BaseRetriever initialization."""
        # Can't instantiate abstract class directly, use concrete implementation
        retriever = GuestRetriever()
        assert isinstance(retriever.docstore, SimpleDocumentStore)
        assert retriever.retriever is None
        assert retriever.documents == []

    @patch('datasets.load_dataset')
    def test_load_success(self, mock_load_dataset):
        """Test successful loading and transformation of dataset."""
        # Mock dataset
        mock_dataset = [
            {
                'name': 'Test Guest',
                'relation': 'Friend',
                'description': 'A test guest',
                'email': 'test@example.com'
            }
        ]
        mock_load_dataset.return_value = mock_dataset

        retriever = GuestRetriever()
        documents = retriever.load("test-dataset")

        assert len(documents) == 1
        assert isinstance(documents[0], Document)
        assert "Test Guest" in documents[0].get_content()
        assert documents[0].metadata["name"] == "Test Guest"

    @patch('datasets.load_dataset')
    def test_load_empty_dataset(self, mock_load_dataset):
        """Test loading empty dataset."""
        mock_load_dataset.return_value = []

        retriever = GuestRetriever()
        documents = retriever.load("empty-dataset")

        assert len(documents) == 0
        assert retriever.documents == []

    def test_search_before_initialize(self):
        """Test search fails before initialization."""
        retriever = GuestRetriever()

        with pytest.raises(ValueError, match="No documents indexed"):
            retriever.search("test query")

    def test_format_NodeWithScores_empty(self):
        """Test formatting empty results."""
        result = BaseRetriever.format_NodeWithScores([], "detailed")
        assert result == "No relevant guests found."

    def test_format_NodeWithScores_simple(self):
        """Test simple formatting."""
        # Create mock NodeWithScore
        mock_node = Mock()
        mock_node.metadata = {"name": "Test Guest"}
        mock_result = Mock()
        mock_result.node = mock_node

        result = BaseRetriever.format_NodeWithScores([mock_result], "simple")
        assert "1. Test Guest" in result

    def test_format_NodeWithScores_detailed(self):
        """Test detailed formatting."""
        # Create mock NodeWithScore
        mock_node = Mock()
        mock_node.get_content.return_value = "Name: Test Guest\nDescription: Test"
        mock_result = Mock()
        mock_result.node = mock_node

        result = BaseRetriever.format_NodeWithScores([mock_result], "detailed")
        assert "Result 1:" in result
        assert "Test Guest" in result

    def test_format_NodeWithScores_invalid_type(self):
        """Test invalid format type raises error."""
        mock_result = Mock()

        with pytest.raises(ValueError, match="Unknown format_type"):
            BaseRetriever.format_NodeWithScores([mock_result], "invalid")


class TestGuestRetriever:
    """Test the BM25 implementation."""

    @patch('datasets.load_dataset')
    @patch('app.infrastructure.retriever.BM25Retriever')
    def test_initialize_success(self, mock_bm25, mock_load_dataset):
        """Test successful initialization."""
        # Mock dataset
        mock_dataset = [
            {
                'name': 'Test Guest',
                'relation': 'Friend',
                'description': 'A test guest',
                'email': 'test@example.com'
            }
        ]
        mock_load_dataset.return_value = mock_dataset

        # Mock BM25Retriever
        mock_retriever_instance = Mock()
        mock_bm25.from_defaults.return_value = mock_retriever_instance

        retriever = GuestRetriever()
        retriever.initialize("test-dataset")

        # Verify initialization
        assert len(retriever.documents) == 1
        assert retriever.retriever == mock_retriever_instance
        mock_bm25.from_defaults.assert_called_once()

    @patch('datasets.load_dataset')
    def test_initialize_indexing_mismatch(self, mock_load_dataset):
        """Test initialization fails with indexing mismatch."""
        mock_dataset = [
            {'name': 'Test', 'relation': 'Friend', 'description': 'Test', 'email': 'test@test.com'}
        ]
        mock_load_dataset.return_value = mock_dataset

        retriever = GuestRetriever()

        # Mock _create_index to return wrong count
        retriever._create_index = Mock(return_value=0)

        with pytest.raises(RuntimeError, match="Indexing mismatch"):
            retriever.initialize("test-dataset")

    def test_perform_search_not_initialized(self):
        """Test search fails when retriever not initialized."""
        retriever = GuestRetriever()

        with pytest.raises(ValueError, match="Retriever not initialized"):
            retriever._perform_search("test query")

    @patch('datasets.load_dataset')
    @patch('app.infrastructure.retriever.BM25Retriever')
    def test_search_success(self, mock_bm25, mock_load_dataset):
        """Test successful search."""
        # Setup mocks
        mock_dataset = [
            {'name': 'Test', 'relation': 'Friend', 'description': 'Test', 'email': 'test@test.com'}
        ]
        mock_load_dataset.return_value = mock_dataset

        mock_retriever_instance = Mock()
        mock_node_result = Mock()
        mock_retriever_instance.retrieve.return_value = [mock_node_result]
        mock_bm25.from_defaults.return_value = mock_retriever_instance

        retriever = GuestRetriever()
        retriever.initialize("test-dataset")

        results = retriever.search("test query")

        assert len(results) == 1
        assert results[0] == mock_node_result
        mock_retriever_instance.retrieve.assert_called_once_with("test query")


class TestMockVectorRetriever:
    """Test the mock vector implementation."""

    def test_init(self):
        """Test MockVectorRetriever initialization."""
        retriever = MockVectorRetriever(similarity_top_k=5)
        assert retriever.similarity_top_k == 5
        assert hasattr(retriever, 'is_indexed')

    @patch('datasets.load_dataset')
    def test_initialize_and_search(self, mock_load_dataset):
        """Test initialization and mock search."""
        # Mock dataset
        mock_dataset = [
            {
                'name': 'Math Professor',
                'relation': 'Colleague',
                'description': 'Expert in mathematics and algorithms',
                'email': 'math@example.com'
            },
            {
                'name': 'Art Teacher',
                'relation': 'Friend',
                'description': 'Teaches painting and drawing',
                'email': 'art@example.com'
            }
        ]
        mock_load_dataset.return_value = mock_dataset

        retriever = MockVectorRetriever(similarity_top_k=2)
        retriever.initialize("test-dataset")

        # Test search
        results = retriever.search("mathematics")

        # Should find the math professor with higher score
        assert len(results) <= 2
        if results:
            assert isinstance(results[0], NodeWithScore)
            # Math professor should score higher for "mathematics" query
            top_result_content = results[0].node.get_content()
            assert "Math Professor" in top_result_content

    def test_search_empty_documents(self):
        """Test search with no documents."""
        retriever = MockVectorRetriever()
        retriever.is_indexed = True
        retriever.documents = []

        results = retriever.search("test")
        assert results == []

    def test_search_not_indexed(self):
        """Test search fails when not indexed."""
        retriever = MockVectorRetriever()

        with pytest.raises(ValueError, match="No documents indexed"):
            retriever.search("test")

    @patch('datasets.load_dataset')
    def test_scoring_algorithm(self, mock_load_dataset):
        """Test the mock scoring algorithm works correctly."""
        mock_dataset = [
            {
                'name': 'Perfect Match',
                'relation': 'Friend',
                'description': 'This contains test query words exactly',
                'email': 'perfect@example.com'
            },
            {
                'name': 'Partial Match',
                'relation': 'Friend',
                'description': 'This contains test but not all words',
                'email': 'partial@example.com'
            },
            {
                'name': 'No Match',
                'relation': 'Friend',
                'description': 'This contains nothing relevant',
                'email': 'nomatch@example.com'
            }
        ]
        mock_load_dataset.return_value = mock_dataset

        retriever = MockVectorRetriever(similarity_top_k=3)
        retriever.initialize("test-dataset")

        results = retriever.search("test query")

        # Should return results sorted by score (highest first)
        assert len(results) >= 1  # At least one match

        # Perfect match should score higher than partial match
        scores = [result.score for result in results if result.score is not None]
        assert scores == sorted(scores, reverse=True)  # Descending order


@pytest.fixture
def sample_documents():
    """Fixture providing sample documents for testing."""
    return [
        Document(
            text=(
                "Name: Alice Smith\nRelation: Friend\n"
                "Description: Software engineer\nEmail: alice@example.com"
            ),
            metadata={"name": "Alice Smith"}
        ),
        Document(
            text=(
                "Name: Bob Jones\nRelation: Colleague\n"
                "Description: Data scientist\nEmail: bob@example.com"
            ),
            metadata={"name": "Bob Jones"}
        )
    ]


class TestIntegration:
    """Integration tests for the retriever system."""

    def test_retriever_swapping(self, sample_documents):
        """Test that different retrievers can be used interchangeably."""
        def test_retriever_interface(retriever: RetrieverInterface):
            """Helper function that works with any retriever."""
            # This should work with any implementation
            assert hasattr(retriever, 'initialize')
            assert hasattr(retriever, 'search')
            return True

        # Test with different implementations
        bm25_retriever = GuestRetriever()
        vector_retriever = MockVectorRetriever()

        assert test_retriever_interface(bm25_retriever)
        assert test_retriever_interface(vector_retriever)

    @patch('datasets.load_dataset')
    def test_end_to_end_workflow(self, mock_load_dataset):
        """Test complete workflow from initialization to search and formatting."""
        # Mock realistic dataset
        mock_dataset = [
            {
                'name': 'Dr. Einstein',
                'relation': 'Scientist',
                'description': 'Theoretical physicist and mathematician',
                'email': 'einstein@princeton.edu'
            }
        ]
        mock_load_dataset.return_value = mock_dataset

        # Test with MockVectorRetriever (no external dependencies)
        retriever = MockVectorRetriever()

        # Initialize
        retriever.initialize("test-dataset")

        # Search
        results = retriever.search("physicist")

        # Format results
        formatted_simple = BaseRetriever.format_NodeWithScores(results, "simple")
        formatted_detailed = BaseRetriever.format_NodeWithScores(results, "detailed")

        # Verify end-to-end functionality
        assert "Dr. Einstein" in formatted_simple
        assert "Dr. Einstein" in formatted_detailed
        assert "physicist" in formatted_detailed.lower()
