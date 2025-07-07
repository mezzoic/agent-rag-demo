"""
Unit tests for the tools infrastructure.
Tests the LlamaIndex tool integration functionality.
"""
from unittest.mock import Mock, patch

import pytest
from llama_index.core.schema import NodeWithScore
from llama_index.core.tools.types import ToolMetadata, ToolOutput

from app.infrastructure.retriever import RetrieverInterface
from app.infrastructure.tools import QueryTool


class TestQueryTool:
    """Test the QueryTool implementation."""

    def test_init(self):
        """Test QueryTool initialization."""
        mock_retriever = Mock(spec=RetrieverInterface)
        tool = QueryTool(retriever=mock_retriever)

        assert tool.retriever == mock_retriever

    def test_metadata(self):
        """Test tool metadata is correct."""
        mock_retriever = Mock(spec=RetrieverInterface)
        tool = QueryTool(retriever=mock_retriever)

        metadata = tool.metadata
        assert isinstance(metadata, ToolMetadata)
        assert metadata.name == "query_guests"
        assert "Search for guests" in metadata.description
        assert "natural language queries" in metadata.description

    def test_call_success(self):
        """Test successful tool execution."""
        # Mock retriever
        mock_retriever = Mock(spec=RetrieverInterface)
        mock_node = Mock()
        mock_node.get_content.return_value = "Name: Test Guest\nDescription: Test person"
        mock_result = Mock(spec=NodeWithScore)
        mock_result.node = mock_node
        mock_retriever.search.return_value = [mock_result]

        # Test tool
        tool = QueryTool(retriever=mock_retriever)
        output = tool("test query")

        # Verify
        assert isinstance(output, ToolOutput)
        assert output.tool_name == "query_guests"
        assert "Test Guest" in output.content
        mock_retriever.search.assert_called_once_with("test query")

    def test_call_no_results(self):
        """Test tool execution with no results."""
        # Mock retriever with no results
        mock_retriever = Mock(spec=RetrieverInterface)
        mock_retriever.search.return_value = []

        # Test tool
        tool = QueryTool(retriever=mock_retriever)
        output = tool("nonexistent query")

        # Verify
        assert isinstance(output, ToolOutput)
        assert output.tool_name == "query_guests"
        assert "No relevant guests found" in output.content

    def test_run_method(self):
        """Test the run method directly."""
        # Mock retriever
        mock_retriever = Mock(spec=RetrieverInterface)
        mock_result = Mock(spec=NodeWithScore)
        mock_retriever.search.return_value = [mock_result]

        # Test tool
        tool = QueryTool(retriever=mock_retriever)
        results = tool.run("test query")

        # Verify
        assert results == [mock_result]
        mock_retriever.search.assert_called_once_with("test query")

    def test_integration_with_real_retriever(self):
        """Test integration with a real retriever (using mock data)."""
        from app.infrastructure.retriever import MockVectorRetriever

        # Use MockVectorRetriever for testing (no external dependencies)
        with patch('datasets.load_dataset') as mock_load_dataset:
            mock_dataset = [
                {
                    'name': 'Test Guest',
                    'relation': 'Friend',
                    'description': 'A wonderful test person',
                    'email': 'test@example.com'
                }
            ]
            mock_load_dataset.return_value = mock_dataset

            # Initialize retriever and tool
            retriever = MockVectorRetriever()
            retriever.initialize("test-dataset")
            tool = QueryTool(retriever=retriever)

            # Test tool execution
            output = tool("test")

            # Verify integration
            assert isinstance(output, ToolOutput)
            assert "Test Guest" in output.content

    def test_error_handling(self):
        """Test tool behavior when retriever raises an error."""
        # Mock retriever that raises an error
        mock_retriever = Mock(spec=RetrieverInterface)
        mock_retriever.search.side_effect = ValueError("Retriever error")

        # Test tool
        tool = QueryTool(retriever=mock_retriever)

        # Should propagate the error
        with pytest.raises(ValueError, match="Retriever error"):
            tool("test query")

    def test_call_with_different_format_types(self):
        """Test tool with different result formatting."""
        # Mock retriever
        mock_retriever = Mock(spec=RetrieverInterface)
        mock_node = Mock()
        mock_node.get_content.return_value = "Name: Test Guest\nDescription: Test"
        mock_node.metadata = {"name": "Test Guest"}
        mock_result = Mock(spec=NodeWithScore)
        mock_result.node = mock_node
        mock_retriever.search.return_value = [mock_result]

        # Test tool (uses detailed format by default)
        tool = QueryTool(retriever=mock_retriever)
        output = tool("test query")

        # Should use detailed format by default
        assert "Result 1:" in output.content
        assert "Test Guest" in output.content

    @patch('app.infrastructure.retriever.BaseRetriever.format_NodeWithScores')
    def test_uses_format_method(self, mock_format):
        """Test that tool uses the BaseRetriever format method."""
        mock_format.return_value = "Formatted results"

        # Mock retriever
        mock_retriever = Mock(spec=RetrieverInterface)
        mock_result = Mock(spec=NodeWithScore)
        mock_retriever.search.return_value = [mock_result]

        # Test tool
        tool = QueryTool(retriever=mock_retriever)
        output = tool("test query")

        # Verify format method was called
        mock_format.assert_called_once_with([mock_result], "detailed")
        assert output.content == "Formatted results"


class TestToolIntegration:
    """Integration tests for the tool system."""

    def test_tool_with_multiple_retrievers(self):
        """Test that tool works with different retriever implementations."""
        from app.infrastructure.retriever import MockVectorRetriever

        with patch('datasets.load_dataset') as mock_load_dataset:
            mock_dataset = [
                {
                    'name': 'Alice',
                    'relation': 'Friend',
                    'description': 'Software engineer',
                    'email': 'alice@example.com'
                },
                {
                    'name': 'Bob',
                    'relation': 'Colleague',
                    'description': 'Data scientist',
                    'email': 'bob@example.com'
                }
            ]
            mock_load_dataset.return_value = mock_dataset

            # Test with MockVectorRetriever
            vector_retriever = MockVectorRetriever()
            vector_retriever.initialize("test-dataset")
            vector_tool = QueryTool(retriever=vector_retriever)

            # Test search
            output = vector_tool("engineer")
            assert isinstance(output, ToolOutput)
            # Should find Alice (software engineer)
            assert "Alice" in output.content or "engineer" in output.content.lower()

    def test_tool_metadata_consistency(self):
        """Test that tool metadata is consistent across instances."""
        mock_retriever1 = Mock(spec=RetrieverInterface)
        mock_retriever2 = Mock(spec=RetrieverInterface)

        tool1 = QueryTool(retriever=mock_retriever1)
        tool2 = QueryTool(retriever=mock_retriever2)

        # Metadata should be the same regardless of retriever
        assert tool1.metadata.name == tool2.metadata.name
        assert tool1.metadata.description == tool2.metadata.description

    def test_end_to_end_workflow(self):
        """Test complete workflow: initialize -> search -> format -> output."""
        from app.infrastructure.retriever import MockVectorRetriever

        with patch('datasets.load_dataset') as mock_load_dataset:
            # Mock dataset with scientific guests
            mock_dataset = [
                {
                    'name': 'Dr. Marie Curie',
                    'relation': 'Scientist',
                    'description': 'Nobel Prize winner in Physics and Chemistry',
                    'email': 'marie@science.edu'
                },
                {
                    'name': 'Albert Einstein',
                    'relation': 'Physicist',
                    'description': 'Theoretical physicist, theory of relativity',
                    'email': 'albert@princeton.edu'
                }
            ]
            mock_load_dataset.return_value = mock_dataset

            # Complete workflow
            retriever = MockVectorRetriever(similarity_top_k=2)
            retriever.initialize("scientists-dataset")
            tool = QueryTool(retriever=retriever)

            # Search for physics-related terms
            output = tool("physics relativity")

            # Verify end-to-end functionality
            assert isinstance(output, ToolOutput)
            assert output.tool_name == "query_guests"
            # Should find Einstein due to physics/relativity terms
            content_lower = output.content.lower()
            assert "einstein" in content_lower or "physics" in content_lower
