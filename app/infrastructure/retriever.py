from abc import ABC, abstractmethod
from typing import Protocol

import datasets
from llama_index.core.schema import Document, NodeWithScore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever


class RetrieverInterface(Protocol):
    """Interface for all retriever implementations."""

    def initialize(self, dataset_name: str = "agents-course/unit3-invitees") -> None:
        """Initialize the retriever with data from the specified dataset."""
        ...

    def search(self, query: str) -> list[NodeWithScore]:
        """Search for relevant documents based on the query."""
        ...


class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""

    def __init__(self):
        self.docstore = SimpleDocumentStore()
        self.retriever = None
        self.documents = []

    @abstractmethod
    def _create_index(self) -> int:
        """Create the search index. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _perform_search(self, query: str) -> list[NodeWithScore]:
        """Perform the actual search. Must be implemented by subclasses."""
        pass

    def load(self, dataset_name: str = "agents-course/unit3-invitees") -> list[Document]:
        """
        Extract and Transform phases:
        - Extract: Load data from Hugging Face dataset
        - Transform: Convert to Document objects
        """
        # Extract phase
        guest_dataset = datasets.load_dataset(dataset_name, split="train")
        print(f"Loaded {len(guest_dataset)} guests from dataset")  # type: ignore

        # Transform phase
        self.documents = [
            Document(
                text="\n".join([
                    f"Name: {guest['name']}",  # type: ignore
                    f"Relation: {guest['relation']}",  # type: ignore
                    f"Description: {guest['description']}",  # type: ignore
                    f"Email: {guest['email']}"  # type: ignore
                ]),
                metadata={"name": guest["name"]}  # type: ignore
            )
            for guest in guest_dataset
        ]

        print(f"Transformed {len(self.documents)} guest records to documents")
        return self.documents

    def initialize(self, dataset_name: str = "agents-course/unit3-invitees") -> None:
        """
        Complete ETL pipeline: calls load() then _create_index().
        Verifies that all loaded documents were successfully indexed.
        """
        documents = self.load(dataset_name)
        indexed_count = self._create_index()

        # Verify all documents were indexed
        if indexed_count != len(documents):
            raise RuntimeError(
                f"Indexing mismatch: loaded {len(documents)} documents "
                f"but indexed {indexed_count}"
            )

        print(f"ETL pipeline complete: {indexed_count} documents loaded and indexed successfully")

    def search(self, query: str) -> list[NodeWithScore]:
        """
        Generic search interface.
        Works with any retriever type that implements retrieve().
        """
        if not self.retriever:
            raise ValueError("No documents indexed. Call initialize() first.")

        return self._perform_search(query)

    @staticmethod
    def format_NodeWithScores(results: list[NodeWithScore], format_type: str = "detailed") -> str:
        """Static method to format search results in different ways."""
        if not results:
            return "No relevant guests found."

        if format_type == "simple":
            formatted = []
            for i, doc in enumerate(results, 1):
                name = doc.node.metadata.get("name", f"Guest {i}")
                formatted.append(f"{i}. {name}")
            return "\n".join(formatted)

        elif format_type == "detailed":
            formatted = []
            for i, doc in enumerate(results, 1):
                content = doc.node.get_content()
                formatted.append(f"Result {i}:\n{content}\n")
            return "\n".join(formatted)

        else:
            raise ValueError(f"Unknown format_type: {format_type}. Use 'simple' or 'detailed'.")


class GuestRetriever(BaseRetriever):
    """BM25-based implementation of the guest retriever."""

    def _create_index(self) -> int:
        """Create BM25 index for keyword-based search."""
        return self.index()

    def _perform_search(self, query: str) -> list[NodeWithScore]:
        """Perform BM25 keyword search."""
        if not self.retriever:
            raise ValueError("Retriever not initialized")
        return self.retriever.retrieve(query)

    def index(self, documents: list[Document] | None = None) -> int:
        """
        Load phase: Index documents for search.
        Uses self.documents if no documents provided.
        """
        docs_to_index = documents if documents is not None else self.documents

        if not docs_to_index:
            raise ValueError("No documents to index. Call load() first or provide documents.")

        # Debug: print document contents
        print(f"Indexing {len(docs_to_index)} documents...")
        for i, doc in enumerate(docs_to_index):
            print(f"Doc {i}: {doc.get_content()[:100]}...")

        # Add documents to the docstore
        for doc in docs_to_index:
            self.docstore.add_documents([doc])

        # Create the BM25 retriever with more verbose output
        try:
            self.retriever = BM25Retriever.from_defaults(
                docstore=self.docstore,
                similarity_top_k=1,
                verbose=True
            )
            print(f"Successfully indexed {len(docs_to_index)} documents for search")
            return len(docs_to_index)
        except Exception as e:
            print(f"Error creating BM25 retriever: {e}")
            raise


# Example of how easy it is to add a new retriever implementation
class MockVectorRetriever(BaseRetriever):
    """
    Mock vector-based retriever to demonstrate extensibility.
    In a real implementation, this would use embeddings/vector similarity.
    """

    def __init__(self, similarity_top_k: int = 3):
        super().__init__()
        self.similarity_top_k = similarity_top_k
        self.is_indexed = False

    def _create_index(self) -> int:
        """Mock vector index creation."""
        print("Creating mock vector embeddings...")
        # In real implementation: generate embeddings, create vector index
        self.is_indexed = True
        return len(self.documents)

    def search(self, query: str) -> list[NodeWithScore]:
        """Override search to bypass the retriever check."""
        if not self.is_indexed:
            raise ValueError("No documents indexed. Call initialize() first.")

        return self._perform_search(query)

    def _perform_search(self, query: str) -> list[NodeWithScore]:
        """Mock vector similarity search."""
        if not self.documents:
            return []

        print(f"Performing mock vector similarity search for: '{query}'")

        # Mock logic: return documents based on simple text similarity
        results = []
        for doc in self.documents:
            # Mock similarity score (in real implementation: compute vector similarity)
            content = doc.get_content().lower()
            query_lower = query.lower()

            # Simple mock scoring: count query word occurrences
            query_words = query_lower.split()
            score = sum(1 for word in query_words if word in content) / len(query_words)

            if score > 0:
                results.append(NodeWithScore(
                    node=doc,
                    score=score
                ))

        # Sort by score (highest first)
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:self.similarity_top_k]
