
from llama_index.core.schema import NodeWithScore
from llama_index.core.tools import BaseTool
from llama_index.core.tools.types import ToolMetadata, ToolOutput

from .retriever import BaseRetriever, RetrieverInterface


class QueryTool(BaseTool):
    """Tool to query the guest database using retrieval."""

    def __init__(self, retriever: RetrieverInterface):
        self.retriever = retriever
        self.retriever.initialize()

    @property
    def metadata(self) -> ToolMetadata:
        """Return tool metadata."""
        return ToolMetadata(
            name="query_guests",
            description=(
                "Search for guests in the database using natural language queries. "
                "Returns relevant guest information."
            ),
        )

    def __call__(self, query: str) -> ToolOutput:
        """Main entry point for the tool - implements the abstract __call__ method."""
        results = self.run(query)

        # Format results as string content using the static method
        formatted_content = BaseRetriever.format_NodeWithScores(results, "detailed")

        return ToolOutput(
            tool_name="query_guests",
            content=formatted_content,
            raw_input={"query": query},  # Store the input query
            raw_output=results  # Store the original results for programmatic access
        )

    def run(self, query: str) -> list[NodeWithScore]:
        """Run the query against the retriever and return results."""
        results = self.retriever.search(query)
        return results

