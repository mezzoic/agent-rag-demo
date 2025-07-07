"""
Demonstration of how easy it is to swap retriever implementations
without changing client code, thanks to dependency inversion.
"""

from infrastructure.retriever import GuestRetriever, MockVectorRetriever, RetrieverInterface


def demonstrate_retriever_swapping():
    """Show how different retrievers can be used interchangeably."""

    # Client function that works with any retriever implementation
    def run_search_demo(retriever: RetrieverInterface, retriever_name: str):
        print(f"\n{'='*50}")
        print(f"Testing {retriever_name}")
        print(f"{'='*50}")

        # Initialize the retriever
        retriever.initialize()

        # Search for the same query
        query = "mathematician computer"
        print(f"\nSearching for: '{query}'")

        results = retriever.search(query)
        print(f"Found {len(results)} results:")

        for i, result in enumerate(results, 1):
            score = result.score if hasattr(result, 'score') else 'N/A'
            name = result.node.metadata.get("name", f"Guest {i}")
            print(f"  {i}. {name} (Score: {score:.3f})")

    # Demonstrate swapping different implementations
    print("Demonstrating Retriever Swapping with Dependency Inversion")
    print("The client code remains exactly the same!")

    # Test BM25 retriever
    bm25_retriever: RetrieverInterface = GuestRetriever()
    run_search_demo(bm25_retriever, "BM25 Keyword Retriever")

    # Test mock vector retriever
    vector_retriever: RetrieverInterface = MockVectorRetriever(similarity_top_k=2)
    run_search_demo(vector_retriever, "Mock Vector Similarity Retriever")

    print(f"\n{'='*50}")
    print("Demo Complete!")
    print("Notice how the client code (run_search_demo function)")
    print("worked with both retrievers without any changes!")
    print("This is the power of dependency inversion.")
    print(f"{'='*50}")

if __name__ == "__main__":
    demonstrate_retriever_swapping()
