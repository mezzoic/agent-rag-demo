"""
RAG Demo without LLM - Shows core retrieval functionality
This version demonstrates the RAG system without requiring any LLM setup.
"""
import asyncio

from infrastructure.retriever import GuestRetriever, RetrieverInterface
from infrastructure.tools import QueryTool


async def main():
    print("=== RAG Demo (No LLM Required) ===")
    print("This demo shows the core retrieval and tool functionality")
    print("without requiring Ollama, OpenAI, or any cloud API keys.\n")

    # Initialize the retriever with complete ETL pipeline
    retriever: RetrieverInterface = GuestRetriever()
    guest_info_tool = QueryTool(retriever)

    # Test queries
    queries = ["mathematician", "physicist", "inventor", "scientist"]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print('='*60)

        # Test tool directly (this is what the LLM would use)
        print("\n--- Tool Output (what LLM would receive) ---")
        tool_result = guest_info_tool(query)
        print(tool_result.content)

        # Show raw retrieval scores for debugging
        print("\n--- Raw Retrieval Results ---")
        raw_results = retriever.search(query)
        print(f"Found {len(raw_results)} results:")
        for i, result in enumerate(raw_results, 1):
            score = result.score if hasattr(result, 'score') else 'N/A'
            name = result.node.metadata.get("name", f"Guest {i}")
            print(f"  {i}. {name} (Relevance Score: {score:.3f})")

    print(f"\n{'='*60}")
    print("✅ RAG System Working Successfully!")
    print("\nTo add LLM capabilities:")
    print("1. Install Ollama: https://ollama.ai/")
    print("   Run: ollama pull llama3.1:8b")
    print("2. Or get a Hugging Face API key: https://huggingface.co/settings/tokens")
    print("   Set: HUGGINGFACE_API_KEY=your_token")
    print("3. Or get an OpenAI API key and set: OPENAI_API_KEY=your_key")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())
