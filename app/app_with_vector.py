"""
Alternative version of app.py showing how easy it is to swap retrievers.
The client code is nearly identical - only the instantiation line changes!
"""

from infrastructure.retriever import MockVectorRetriever, RetrieverInterface

# Initialize the retriever with complete ETL pipeline
# Type annotation uses interface, but instantiation uses concrete class
# Notice: Only this line changed from app.py!
retriever: RetrieverInterface = MockVectorRetriever(similarity_top_k=2)
retriever.initialize()

# Test different formatting options - EXACT SAME CODE as app.py!
query = "mathematician"
print(f"\nQuery: {query}")

print("\n=== Using search() + static formatter ===")
raw_results = retriever.search(query)

print(f"Found {len(raw_results)} results:")
for i, result in enumerate(raw_results, 1):
    score = result.score if hasattr(result, 'score') else 'N/A'
    name = result.node.metadata.get("name", f"Guest {i}")
    print(f"{i}. {name} (Score: {score})")

print("\nDetailed format:")
formatted_detailed = retriever.format_NodeWithScores(raw_results, "detailed")
print(formatted_detailed[:500] + "..." if len(formatted_detailed) > 500 else formatted_detailed)
