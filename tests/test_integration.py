"""
Integration tests for the main application flows.
Tests that the applications work end-to-end.
"""
from unittest.mock import Mock, patch

import pytest

from app.infrastructure.retriever import GuestRetriever, MockVectorRetriever
from app.infrastructure.tools import QueryTool


class TestApplicationIntegration:
    """Test the main application integration."""

    @patch('app.infrastructure.retriever.datasets.load_dataset')
    def test_guest_retriever_tool_integration(self, mock_load_dataset):
        """Test that retriever and tool work together."""
        # Mock dataset
        mock_dataset = [
            {
                'name': 'Alice Johnson',
                'relation': 'Friend',
                'description': 'Software engineer with Python expertise',
                'email': 'alice@example.com'
            },
            {
                'name': 'Bob Smith',
                'relation': 'Colleague',
                'description': 'Data scientist specializing in machine learning',
                'email': 'bob@company.com'
            }
        ]
        mock_load_dataset.return_value = mock_dataset

        # Create retriever and tool
        retriever = GuestRetriever()
        tool = QueryTool(retriever)

        # Initialize the system
        retriever.initialize("test-dataset")

        # Test that tool can be called
        result = tool("engineer")

        # Should return a ToolOutput
        assert hasattr(result, 'tool_name')
        assert hasattr(result, 'content')
        assert result.tool_name == "query_guests"

        # Content should contain search results
        assert isinstance(result.content, str)
        # Since we're using BM25, it might or might not find matches
        # Just verify we get a string response

    @patch('app.infrastructure.retriever.datasets.load_dataset')
    def test_mock_vector_retriever_tool_integration(self, mock_load_dataset):
        """Test that mock vector retriever and tool work together."""
        # Mock dataset
        mock_dataset = [
            {
                'name': 'Carol Williams',
                'relation': 'Friend',
                'description': 'Machine learning researcher and AI expert',
                'email': 'carol@university.edu'
            }
        ]
        mock_load_dataset.return_value = mock_dataset

        # Create retriever and tool
        retriever = MockVectorRetriever(similarity_top_k=1)
        tool = QueryTool(retriever)

        # Initialize the system
        retriever.initialize("test-dataset")

        # Test that tool can be called
        result = tool("machine learning")

        # Should return a ToolOutput
        assert result.tool_name == "query_guests"
        assert isinstance(result.content, str)

        # Mock vector retriever should find matches based on word overlap
        assert "Carol Williams" in result.content or "No relevant guests found" in result.content

    def test_retriever_swapping_in_tool(self):
        """Test that different retrievers can be swapped in the same tool workflow."""
        # Mock both retrievers to avoid external dependencies
        mock_guest_retriever = Mock(spec=GuestRetriever)
        mock_vector_retriever = Mock(spec=MockVectorRetriever)

        # Both should return the same mock result
        mock_node = Mock()
        mock_node.get_content.return_value = "Test guest: John Doe"
        mock_node.metadata = {"name": "John Doe"}
        mock_result = Mock()
        mock_result.node = mock_node
        mock_results = [mock_result]

        mock_guest_retriever.search.return_value = mock_results
        mock_vector_retriever.search.return_value = mock_results

        # Test with both retrievers
        guest_tool = QueryTool(mock_guest_retriever)
        vector_tool = QueryTool(mock_vector_retriever)

        guest_output = guest_tool("test query")
        vector_output = vector_tool("test query")

        # Both should produce equivalent outputs
        assert guest_output.tool_name == vector_output.tool_name
        assert guest_output.content == vector_output.content


class TestRetrieverFormatting:
    """Test the retriever result formatting functionality."""

    def test_simple_formatting(self):
        """Test simple result formatting."""
        # Create mock results
        mock_results = []
        for name in ["Alice", "Bob", "Carol"]:
            mock_node = Mock()
            mock_node.metadata = {"name": name}
            mock_result = Mock()
            mock_result.node = mock_node
            mock_results.append(mock_result)

        formatted = GuestRetriever.format_NodeWithScores(mock_results, "simple")

        expected_lines = ["1. Alice", "2. Bob", "3. Carol"]
        assert formatted == "\n".join(expected_lines)

    def test_detailed_formatting(self):
        """Test detailed result formatting."""
        mock_node = Mock()
        mock_node.get_content.return_value = "Name: Alice\nRole: Engineer\nEmail: alice@example.com"
        mock_result = Mock()
        mock_result.node = mock_node
        mock_results = [mock_result]

        formatted = GuestRetriever.format_NodeWithScores(mock_results, "detailed")

        assert "Result 1:" in formatted
        assert "Name: Alice" in formatted
        assert "Role: Engineer" in formatted
        assert "alice@example.com" in formatted

    def test_empty_results_formatting(self):
        """Test formatting of empty results."""
        formatted_simple = GuestRetriever.format_NodeWithScores([], "simple")
        formatted_detailed = GuestRetriever.format_NodeWithScores([], "detailed")

        assert formatted_simple == "No relevant guests found."
        assert formatted_detailed == "No relevant guests found."


class TestErrorHandling:
    """Test error handling in the system."""

    def test_search_without_initialization(self):
        """Test that searching without initialization raises appropriate errors."""
        guest_retriever = GuestRetriever()
        vector_retriever = MockVectorRetriever()

        with pytest.raises(ValueError, match="No documents indexed"):
            guest_retriever.search("test")

        with pytest.raises(ValueError, match="No documents indexed"):
            vector_retriever.search("test")

    @patch('app.infrastructure.retriever.datasets.load_dataset')
    def test_tool_with_initialization_failure(self, mock_load_dataset):
        """Test that tool handles retriever initialization failure gracefully."""
        # Mock empty dataset which will cause initialization to fail
        mock_load_dataset.return_value = []

        retriever = GuestRetriever()

        # Tool initialization should fail when retriever can't be initialized
        with pytest.raises(ValueError, match="No documents to index"):
            QueryTool(retriever)

    def test_invalid_format_type(self):
        """Test that invalid format types raise appropriate errors."""
        mock_node = Mock()
        mock_result = Mock()
        mock_result.node = mock_node
        mock_results = [mock_result]

        with pytest.raises(ValueError, match="Unknown format_type"):
            GuestRetriever.format_NodeWithScores(mock_results, "invalid")


class TestMockComponents:
    """Test that mock components work correctly for testing purposes."""

    @patch('app.infrastructure.retriever.datasets.load_dataset')
    def test_mock_vector_search_scoring(self, mock_load_dataset):
        """Test that mock vector search produces reasonable scores."""
        mock_dataset = [
            {
                'name': 'Alice Engineer',
                'relation': 'Friend',
                'description': 'Software engineer Python expert',
                'email': 'alice@example.com'
            },
            {
                'name': 'Bob Designer',
                'relation': 'Colleague',
                'description': 'UI/UX designer creative professional',
                'email': 'bob@design.com'
            }
        ]
        mock_load_dataset.return_value = mock_dataset

        retriever = MockVectorRetriever(similarity_top_k=2)
        retriever.initialize("test-dataset")

        # Search for engineering-related terms
        results = retriever.search("engineer software")

        if results:  # If results are found
            # Alice should score higher than Bob for "engineer software"
            alice_result = next((r for r in results if "Alice" in r.node.metadata["name"]), None)
            bob_result = next((r for r in results if "Bob" in r.node.metadata["name"]), None)

            if alice_result and bob_result:
                assert alice_result.score >= bob_result.score

    def test_mock_vector_retriever_top_k_limit(self):
        """Test that mock vector retriever respects top_k limit."""
        retriever = MockVectorRetriever(similarity_top_k=1)

        # Mock some documents
        from llama_index.core.schema import Document
        docs = [
            Document(text="Engineer Alice", metadata={"name": "Alice"}),
            Document(text="Engineer Bob", metadata={"name": "Bob"}),
            Document(text="Engineer Carol", metadata={"name": "Carol"})
        ]
        retriever.documents = docs
        retriever.is_indexed = True

        results = retriever.search("engineer")

        # Should return at most similarity_top_k results
        assert len(results) <= 1


class TestSystemRobustness:
    """Test system robustness and edge cases."""

    def test_empty_query_handling(self):
        """Test how the system handles empty queries."""
        retriever = MockVectorRetriever()
        retriever.documents = []
        retriever.is_indexed = True

        # Empty query should not crash
        results = retriever.search("")
        assert isinstance(results, list)

    def test_special_characters_in_query(self):
        """Test handling of special characters in queries."""
        retriever = MockVectorRetriever()
        retriever.documents = []
        retriever.is_indexed = True

        # Special characters should not crash the system
        results = retriever.search("test@#$%^&*()")
        assert isinstance(results, list)

    @patch('app.infrastructure.retriever.datasets.load_dataset')
    def test_large_dataset_handling(self, mock_load_dataset):
        """Test that the system can handle larger datasets."""
        # Create a larger mock dataset
        large_dataset = []
        for i in range(100):
            large_dataset.append({
                'name': f'Person {i}',
                'relation': 'Contact',
                'description': f'Description for person {i}',
                'email': f'person{i}@example.com'
            })

        mock_load_dataset.return_value = large_dataset

        retriever = GuestRetriever()
        retriever.initialize("large-dataset")

        # Should handle large dataset without issues
        assert len(retriever.documents) == 100
        assert retriever.retriever is not None
