# RAG Retrieval System Architecture Guide

## 🏗️ Overview

This project demonstrates a **production-ready, extensible RAG (Retrieval-Augmented Generation) system** built with SOLID principles. It provides a clean foundation for building retrieval systems that can easily evolve from simple keyword search to advanced vector similarity, hybrid retrieval, and LLM-enhanced search.

## 🎯 Architecture Goals

- **Extensible**: Easy to add new retrieval strategies without breaking existing code
- **Testable**: Clean interfaces enable easy mocking and unit testing
- **Maintainable**: Clear separation of concerns and dependency inversion
- **Production-Ready**: Follows enterprise software design patterns
- **Framework-Agnostic**: Can work with LlamaIndex, LangChain, or custom frameworks

## 🧩 Core Components

### 1. Interface Layer (`RetrieverInterface`)
```python
class RetrieverInterface(Protocol):
    def initialize(self, dataset_name: str) -> None: ...
    def search(self, query: str) -> List[NodeWithScore]: ...
```
**Purpose**: Defines the contract that all retrievers must follow
**Benefit**: Enables dependency inversion - clients depend on this interface, not concrete implementations

### 2. Base Implementation (`BaseRetriever`)
```python
class BaseRetriever(ABC):
    def load(self, dataset_name: str) -> List[Document]: ...
    def initialize(self, dataset_name: str) -> None: ...
    def search(self, query: str) -> List[NodeWithScore]: ...
    # Extension points:
    @abstractmethod
    def _create_index(self) -> int: ...
    @abstractmethod
    def _perform_search(self, query: str) -> List[NodeWithScore]: ...
```
**Purpose**: Provides common ETL (Extract, Transform, Load) pipeline
**Benefit**: New retrievers only need to implement indexing and search logic

### 3. Concrete Implementations
- **`GuestRetriever`**: BM25 keyword-based search
- **`MockVectorRetriever`**: Example vector similarity search
- **Your Future Retrievers**: Hybrid, semantic, LLM-enhanced, etc.

### 4. Tool Integration (`QueryTool`)
```python
class QueryTool(BaseTool):
    def __init__(self, retriever: RetrieverInterface): ...
    def __call__(self, query: str) -> ToolOutput: ...
```
**Purpose**: Wraps retrievers for LlamaIndex agent/workflow integration
**Benefit**: Makes your retrievers usable in agent frameworks

## 🎨 SOLID Principles Implementation

### 🔹 Single Responsibility Principle (SRP)
Each class has **one reason to change**:
- `BaseRetriever`: ETL pipeline management
- `GuestRetriever`: BM25-specific indexing and search
- `QueryTool`: Tool interface adaptation
- Data models: Guest representation only

### 🔹 Open/Closed Principle (OCP)
The system is **open for extension, closed for modification**:
```python
# Adding new retriever: NO client code changes needed
class SemanticRetriever(BaseRetriever):
    def _create_index(self) -> int:
        # Your vector index implementation
        pass
    
    def _perform_search(self, query: str) -> List[NodeWithScore]:
        # Your semantic search implementation
        pass

# Client code unchanged:
retriever: RetrieverInterface = SemanticRetriever()  # Only this line changes!
```

### 🔹 Liskov Substitution Principle (LSP)
Any retriever can **replace any other retriever** without breaking functionality:
```python
def test_any_retriever(retriever: RetrieverInterface):
    retriever.initialize()
    results = retriever.search("test query")
    assert len(results) >= 0  # Works with ANY retriever implementation
```

### 🔹 Interface Segregation Principle (ISP)
Clients only depend on **methods they actually use**:
- `RetrieverInterface`: Only search methods
- `ToolMetadata`: Only tool description
- Clean, focused interfaces

### 🔹 Dependency Inversion Principle (DIP)
High-level modules depend on **abstractions, not concretions**:
```python
# app.py - depends on interface, not implementation
retriever: RetrieverInterface = GuestRetriever()  # ✅ Interface type
# NOT: retriever = GuestRetriever()                # ❌ Concrete type
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
# Core dependencies
pip install llama-index datasets llama-index-retrievers-bm25

# For Hugging Face integration (optional)
pip install llama-index-llms-huggingface
```

### 2. Basic Usage
```python
from app.infrastructure.retriever import GuestRetriever

# Initialize retriever
retriever = GuestRetriever()
retriever.initialize()

# Search
results = retriever.search("mathematician")
formatted = GuestRetriever.format_NodeWithScores(results, "detailed")
print(formatted)
```

### 3. Tool Usage (for Agents)
```python
from app.infrastructure.tools import QueryTool
from app.infrastructure.retriever import GuestRetriever

# Create tool
tool = QueryTool(GuestRetriever())

# Use with agents/workflows
result = tool("find mathematicians")
print(result.content)  # Formatted output
print(result.raw_output)  # Raw search results
```

## 🔧 Extending the System

### Adding a New Retriever

1. **Create your retriever class**:
```python
class MyCustomRetriever(BaseRetriever):
    def __init__(self, custom_param: str = "default"):
        super().__init__()
        self.custom_param = custom_param
    
    def _create_index(self) -> int:
        """Implement your indexing logic"""
        # Example: Create vector embeddings, build FAISS index, etc.
        return len(self.documents)
    
    def _perform_search(self, query: str) -> List[NodeWithScore]:
        """Implement your search logic"""
        # Example: Vector similarity, neural search, etc.
        return search_results
```

2. **Use it immediately**:
```python
# No other code changes needed!
retriever: RetrieverInterface = MyCustomRetriever(custom_param="value")
retriever.initialize()
results = retriever.search("query")
```

### Adding New Data Sources

Modify the `load()` method in `BaseRetriever` or override it in your custom retriever:
```python
def load(self, data_source: str) -> List[Document]:
    """Override to support new data sources"""
    if data_source.startswith("http"):
        # Load from API
        pass
    elif data_source.endswith(".csv"):
        # Load from CSV
        pass
    # etc.
```

### Integration Examples

#### With LangChain
```python
# Convert to LangChain tool
langchain_tool = tool.to_langchain_tool()
```

#### With Custom Agents
```python
class MyAgent:
    def __init__(self, retriever: RetrieverInterface):
        self.retriever = retriever  # Works with any retriever!
    
    def process_query(self, query: str):
        results = self.retriever.search(query)
        # Your agent logic here
```

## 🧪 Testing Strategy

### Unit Testing Retrievers
```python
def test_retriever_contract():
    """Test that any retriever follows the interface contract"""
    retrievers = [GuestRetriever(), MockVectorRetriever(), MyCustomRetriever()]
    
    for retriever in retrievers:
        retriever.initialize()
        results = retriever.search("test")
        assert isinstance(results, list)
        # More contract tests...
```

### Mocking for Integration Tests
```python
class MockRetriever:
    def search(self, query: str):
        return [mock_result]  # Predictable test data

def test_my_app():
    app = MyApp(MockRetriever())  # Dependency injection
    # Test app logic without real retrieval
```

## 📊 Performance Considerations

### Indexing Performance
- **Lazy Loading**: Initialize retrievers only when needed
- **Caching**: Cache indexes between runs
- **Batch Processing**: Load documents in batches for large datasets

### Search Performance
- **Connection Pooling**: For database-backed retrievers
- **Result Caching**: Cache frequent queries
- **Async Operations**: Use async versions for I/O-bound operations

## 🔮 Future Extensions

This architecture makes these extensions straightforward:

### Vector Search
```python
class VectorRetriever(BaseRetriever):
    def _create_index(self):
        # Use sentence-transformers, OpenAI embeddings, etc.
        # Build FAISS, Pinecone, Weaviate index
        
    def _perform_search(self, query: str):
        # Vector similarity search
```

### Hybrid Retrieval
```python
class HybridRetriever(BaseRetriever):
    def __init__(self, keyword_weight=0.5, vector_weight=0.5):
        self.bm25_retriever = GuestRetriever()
        self.vector_retriever = VectorRetriever()
        
    def _perform_search(self, query: str):
        # Combine BM25 + vector results with weights
```

### LLM-Enhanced Retrieval
```python
class LLMRetriever(BaseRetriever):
    def _perform_search(self, query: str):
        # 1. Initial retrieval
        # 2. LLM query expansion/rewriting
        # 3. Re-rank with LLM
        # 4. Return enhanced results
```

### Multi-Modal Search
```python
class MultiModalRetriever(BaseRetriever):
    def _perform_search(self, query: str):
        # Search text, images, audio, video
        # Cross-modal similarity
```

## 🎯 Best Practices

1. **Always use interface types** in client code
2. **Initialize retrievers early** (they're designed for reuse)
3. **Handle empty results gracefully** (all retrievers can return [])
4. **Use static formatting methods** for consistent output
5. **Test with multiple retriever implementations** to ensure portability
6. **Profile performance** with your specific data and queries
7. **Consider async patterns** for high-throughput scenarios

## 📁 Project Structure

```
app/
├── app.py                          # Example: BM25 retriever client
├── app_with_vector.py             # Example: Vector retriever client  
├── demo_swappable_retrievers.py   # Demo: Shows easy swapping
└── infrastructure/
    ├── retriever.py               # Core: All retriever classes
    └── tools.py                   # Integration: LlamaIndex tool wrapper
```

## 🤝 Contributing

When adding new retrievers:
1. Inherit from `BaseRetriever`
2. Implement `_create_index()` and `_perform_search()`
3. Add type hints and docstrings
4. Test with existing client code to ensure compatibility
5. Update this documentation with your new retriever example

---

**This architecture provides a solid foundation for any RAG system. Start with the simple BM25 implementation, then evolve to sophisticated vector search, hybrid retrieval, or LLM-enhanced search - all without breaking your existing application code!** 🚀
