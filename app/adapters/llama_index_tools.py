"""
LlamaIndex Tool Adapter - Adapts our clean architecture to LlamaIndex tools.
"""
from llama_index.core.tools import BaseTool
from llama_index.core.tools.types import ToolMetadata, ToolOutput

from ..application.use_cases import RAGWorkflowUseCase


class CleanRAGTool(BaseTool):
    """LlamaIndex tool that uses clean architecture."""

    def __init__(self, rag_workflow: RAGWorkflowUseCase):
        self._rag_workflow = rag_workflow

    @property
    def metadata(self) -> ToolMetadata:
        """Return tool metadata."""
        return ToolMetadata(
            name="search_guests",
            description=(
                "Search for guests using natural language queries. "
                "Returns relevant guest information with clean architecture principles."
            ),
        )

    def __call__(self, query: str) -> ToolOutput:
        """Execute the tool using clean architecture."""
        # Use case orchestrates the entire workflow
        formatted_results = self._rag_workflow.execute_query(query)

        return ToolOutput(
            tool_name="search_guests",
            content=formatted_results,
            raw_input={"query": query},
            raw_output={"formatted_results": formatted_results}
        )
