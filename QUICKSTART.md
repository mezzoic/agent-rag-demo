# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Install and Run
```bash
# Clone/download the project
cd agent-rag-demo

# Install dependencies  
pip install -r requirements.txt

# Run the basic example
python app/app.py
```

### 2. Try Different Retrievers
```bash
# BM25 keyword search
python app/app.py

# Mock vector search  
python app/app_with_vector.py

# Compare both side-by-side
python app/demo_swappable_retrievers.py
```

### 3. Understand the Magic
The key insight: **Only one line changes** between different retrieval strategies!

```python
# app.py - BM25 search
retriever: RetrieverInterface = GuestRetriever()

# app_with_vector.py - Vector search  
retriever: RetrieverInterface = MockVectorRetriever()

# Everything else stays exactly the same!
```

## 🔧 Common Use Cases

### Use Case 1: Add Vector Search
```python
# 1. Create your vector retriever
class RealVectorRetriever(BaseRetriever):
    def _create_index(self) -> int:
        # Use sentence-transformers or OpenAI
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode([doc.get_content() for doc in self.documents])
        return len(self.documents)
    
    def _perform_search(self, query: str) -> List[NodeWithScore]:
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        # Return top results...

# 2. Use it (no other changes needed!)
retriever: RetrieverInterface = RealVectorRetriever()
```

### Use Case 2: Add Your Own Data
```python
# Override the load method
class MyDataRetriever(BaseRetriever):
    def load(self, data_source: str) -> List[Document]:
        # Load from your database, API, files, etc.
        my_data = load_from_database(data_source)
        return [Document(text=item.content, metadata=item.metadata) 
                for item in my_data]

# Use exactly like before
retriever = MyDataRetriever()
retriever.initialize("my_database_connection_string")
```

### Use Case 3: Integration with Agents
```python
from app.infrastructure.tools import QueryTool

# Create tool with any retriever
tool = QueryTool(MyCustomRetriever())

# Use with LlamaIndex agents
from llama_index.core.agent import ReActAgent
agent = ReActAgent.from_tools([tool])

response = agent.chat("Find all mathematicians in the guest list")
print(response)
```

## 🎯 Key Benefits for Real Projects

✅ **Start Simple**: Begin with BM25, upgrade to vectors later  
✅ **Zero Downtime**: Swap retrievers without redeploying  
✅ **A/B Testing**: Compare retrieval strategies easily  
✅ **Team Development**: Multiple devs can work on different retrievers  
✅ **Future-Proof**: Ready for new retrieval techniques  

## 📚 Next Steps

1. **Read**: [ARCHITECTURE.md](ARCHITECTURE.md) for deep dive
2. **Experiment**: Try adding your own retriever
3. **Integrate**: Use with your existing agent/workflow
4. **Extend**: Add new data sources, retrieval methods

## 🤝 Need Help?

The architecture is designed to be self-documenting:
- Interface defines what you must implement
- Base class provides common functionality  
- Examples show the patterns
- Everything follows SOLID principles

**Happy building!** 🎉
