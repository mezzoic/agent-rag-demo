
import random

from huggingface_hub import list_models
from llama_index.core.schema import NodeWithScore
from llama_index.core.tools import BaseTool
from llama_index.core.tools.types import ToolMetadata, ToolOutput
from llama_index.tools.google import GoogleSearchToolSpec

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

class WebSearchTool(BaseTool):
    """Web search tool using DuckDuckGo."""

    def __init__(self, google_api_key: str, google_search_engine_id: str):
        # Initialize any web-specific resources if needed
        try:
            self.search_tool = GoogleSearchToolSpec(key=google_api_key,
                                                engine=google_search_engine_id)
        except Exception as e:
            raise Exception(f"Failed to initialize web search tool: {str(e)}") from e

    @property
    def metadata(self) -> ToolMetadata:
        """Return tool metadata."""
        return ToolMetadata(
            name="web_search",
            description="Search the web using Google custom search api for current information."
        )

    def __call__(self, query: str) -> ToolOutput:
        """Main entry point for the tool - implements the abstract __call__ method."""
        # Placeholder implementation
        print(f"Running web search with query: {query}")
        try:
            result = self.run(query)
            print(f"Web search result: {result}")
            return ToolOutput(
                tool_name="web_search_tool",
                content=result,
                raw_input={"query": query},
                raw_output=result
            )
        except Exception as e:
            print(f"Error during web search: {str(e)}")
            return ToolOutput(
                tool_name="web_search_tool",
                content=f"Error occurred: {str(e)}",
                raw_input={"query": query},
                raw_output=None
            )

    def run(self, query: str) -> str:
        """Run the web tool with the given query."""
        try:
            result = self.search_tool.google_search(query)
            return str(result)
        except Exception as e:
            return f"Error running web tool: {str(e)}"

class WeatherInfoTool(BaseTool):
    """Fetches weather information for any given location"""

    def __init__(self):
        # Initialize any weather-specific resources if needed
        self.weather_conditions = [
            {"condition": "sunny", "temperature": "25°C"},
            {"condition": "rainy", "temperature": "18°C"},
            {"condition": "cloudy", "temperature": "20°C"},
            {"condition": "snowy", "temperature": "-5°C"},
            {"condition": "stormy", "temperature": "-10°C"},
            {"condition": "foggy", "temperature": "-3°C"},
            {"condition": "hazy", "temperature": "-7°C"},
            {"condition": "windy", "temperature": "15°C"},
            {"condition": "thunderstorm", "temperature": "22°C"},
            {"condition": "hail", "temperature": "10°C"}
        ]
    @property
    def metadata(self) -> ToolMetadata:
        """Return tool metadata."""
        return ToolMetadata(
            name="weather_info",
            description="Get current weather information for a specified location."
        )

    def __call__(self, location: str) -> ToolOutput:
        """Main entry point for the tool - implements the abstract __call__ method."""
        # Placeholder implementation
        print(f"Getting weather info for: {location}")
        result = self.run(location)
        return ToolOutput(
            tool_name="weather_info_tool",
            content=result,
            raw_input={"location": location},
            raw_output=result
        )
    def run(self, location: str) -> str:
        """Run the weather tool with the given location."""
        # Placeholder implementation
        data = random.choice(self.weather_conditions)
        return f"Weather info for {location}: {data['condition']}, {data['temperature']}"

class HuggingFaceModelSearchTool(BaseTool):
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""

    def __init__(self):
        # Initialize any Hugging Face-specific resources if needed
        pass

    @property
    def metadata(self) -> ToolMetadata:
        """Return tool metadata."""
        return ToolMetadata(
            name="huggingface_model_search",
            description="Search for downloaded models on Hugging Face Hub by author."
        )

    def __call__(self, query: str) -> ToolOutput:
        """Main entry point for the tool - implements the abstract __call__ method."""
        print(f"Searching Hugging Face models with query: {query}")
        result = self.run(query)
        return ToolOutput(
            tool_name="huggingface_model_search_tool",
            content=result,
            raw_input={"query": query},
            raw_output=result
        )

    def run(self, author: str) -> str:
        """Run the model search tool with the given author."""
        try:
            models = list(list_models( author=author, sort="downloads", direction=-1, limit=1))
            if models:
                model = models[0]
                return f"The most downloaded model by {author} "\
                    f"is {model.id} with {model.downloads:,} downloads."
            else:
                return f"No models found for author: {author}"
        except Exception as e:
            return f"Error searching Hugging Face models: {str(e)}"
