# RAG Demo with Azure AI Foundry

A clean, extensible RAG (Retrieval-Augmented Generation) system built with **Azure AI Foundry**, **LlamaIndex**, and **Hugging Face datasets**. Features SOLID design principles, multiple LLM providers, and enterprise-ready deployment.

## 🚀 Quick Start (5 minutes)

### Option 1: Cloud-First Setup (Recommended)

1. **Clone and setup:**
   ```bash
   git clone <your-repo>
   cd agent-rag-demo
   ```

2. **Configure Azure AI:**
   ```bash
   cp .env.example .env
   # Edit .env with your Azure AI Foundry credentials
   ```

3. **Run:**
   ```bash
   uv run app/app.py
   ```

### Option 2: Local Development

1. **Install Ollama:**
   ```bash
   # Install from https://ollama.ai/
   ollama pull llama3.1:8b
   ```

2. **Run:**
   ```bash
   uv run app/app.py
   ```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Azure AI Foundry (Primary - Cloud)
AZURE_AI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_AI_API_KEY=your-api-key-here
AZURE_AI_MODEL=gpt-4o-mini

# Alternative Providers (Optional Fallbacks)
HUGGINGFACE_API_KEY=your-hf-token-here
OPENAI_API_KEY=your-openai-key-here
```

### LLM Priority Order

The system tries LLMs in this intelligent fallback order:
1. **Azure AI Foundry** (cloud) - Enterprise-grade, if credentials provided
2. **Ollama** (local) - Privacy-first, if installed and running  
3. **Hugging Face** (cloud) - Open-source models, if API key provided
4. **OpenAI** (cloud) - Commercial models, if API key provided
5. **No LLM Mode** - Retriever and tools still work perfectly!

### Azure AI Foundry Setup

1. Go to [Azure AI Foundry](https://ai.azure.com/)
2. Create or select a project
3. Deploy a model (e.g., `gpt-4o-mini`, `gpt-4`)
4. Get your endpoint URL and API key from **Keys and Endpoint**
5. Add them to your `.env` file
6. Use your **deployment name** as the model (not the base model name)

## 🏗️ Architecture

This RAG system follows **SOLID principles** for maximum extensibility:

### Core Components

- **🔍 Retriever Interface**: Clean abstraction for different search methods
- **🛠️ Tool Integration**: LlamaIndex agents with function calling
- **🤖 Multi-LLM Support**: Azure AI, Ollama, Hugging Face, OpenAI
- **📊 BM25 Search**: Fast, relevant document retrieval
- **⚡ Async Workflows**: Non-blocking operations

### File Structure

```
app/
├── app.py                 # Main application with LLM fallbacks
├── app_no_llm.py         # Pure retrieval demo (no LLM needed)
├── infrastructure/
│   ├── retriever.py      # Retrieval interfaces and implementations
│   └── tools.py          # LlamaIndex tool definitions
test_azure.py             # Azure AI connection testing
.env                      # Environment configuration
pyproject.toml           # Project dependencies
```

## 🎯 Features

### ✅ Production Ready
- **Enterprise LLMs**: Azure AI Foundry integration
- **Robust Fallbacks**: Multiple LLM providers with smart switching
- **Error Handling**: Graceful degradation when services unavailable
- **Environment Config**: Secure credential management with `.env`

### ✅ Developer Friendly  
- **SOLID Design**: Easy to extend with new retrievers/LLMs
- **Type Safety**: Full type annotations with protocols
- **Clean Interfaces**: Abstract base classes for extensibility
- **Comprehensive Testing**: Connection tests and validation

### ✅ Flexible Deployment
- **Cloud-First**: Azure AI for production workloads
- **Local Development**: Ollama for offline work
- **Hybrid Mode**: Automatic fallback between cloud/local
- **No-LLM Mode**: Retrieval works independently

## 🧪 Testing

Test your Azure AI connection:
```bash
uv run test_azure.py
```

Test different retrieval methods:
```bash
uv run app/app_no_llm.py
```

## � Documentation

- **[QUICKSTART.md](QUICKSTART.md)**: 5-minute setup guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Deep dive into SOLID design
- **[CLOUD_SETUP.md](CLOUD_SETUP.md)**: Cloud-only deployment guide

## 🔧 Extending the System

### Add New Retrievers

```python
class MyVectorRetriever(BaseRetriever):
    def search(self, query: str) -> List[NodeWithScore]:
        # Your vector search implementation
        pass
```

### Add New LLM Providers

```python
# Add to app.py fallback chain
if not llm and MY_LLM_AVAILABLE:
    llm = MyLLMProvider(...)
```

### Add New Tools

```python
class MyTool(BaseTool):
    def __call__(self, query: str) -> ToolOutput:
        # Your tool implementation
        pass
```

## 💰 Cost Considerations

- **Azure AI Foundry**: ~$0.15-0.30 per 1K tokens (GPT-4o-mini)
- **Development**: Typically under $1-5/month for testing
- **Ollama**: Free local inference (requires GPU/CPU resources)
- **Hybrid**: Use local for dev, cloud for production

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Follow SOLID principles in your code
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Built with**: Azure AI Foundry • LlamaIndex • Hugging Face • Python 3.12+
GuestRetriever, VectorRetriever  ← Specific implementations
```

## 🔧 Extending for Real Projects

### Add Your Data Source
```python
class MyDatabaseRetriever(BaseRetriever):
    def load(self, connection_string: str) -> List[Document]:
        # Connect to your database, API, files, etc.
        data = your_data_source.query(connection_string)
        return [Document(text=item.content) for item in data]
```

### Add Advanced Search
```python
class SemanticRetriever(BaseRetriever):
    def _create_index(self) -> int:
        # Use sentence-transformers, OpenAI embeddings, etc.
        self.embeddings = create_embeddings(self.documents)
        return len(self.documents)
    
    def _perform_search(self, query: str) -> List[NodeWithScore]:
        # Vector similarity, neural search, etc.
        return semantic_search(query, self.embeddings)
```

### Integration with Agents
```python
from app.infrastructure.tools import QueryTool

# Works with any retriever
tool = QueryTool(YourCustomRetriever())

# Use with LlamaIndex, LangChain, or custom agents
agent = ReActAgent.from_tools([tool])
response = agent.chat("Find relevant information")
```

## 💡 Why This Architecture?

### For Startups
- **Start simple** with BM25, **upgrade incrementally** to vector search
- **Experiment easily** with different retrieval strategies
- **Scale confidently** knowing your application won't break

### For Enterprises
- **Zero-downtime deployments** when changing retrieval strategies
- **A/B testing** different approaches in production
- **Team productivity** - multiple teams can work on different retrievers
- **Future-proof** against new retrieval technologies

### For Researchers
- **Easy experimentation** with new retrieval techniques
- **Reproducible results** with consistent interfaces
- **Quick prototyping** of hybrid approaches

## 🎨 Design Benefits

✅ **Stable Client Interface** - Adding new retrievers doesn't break existing code  
✅ **Easy Testing** - Clean interfaces enable mocking and unit tests  
✅ **Clear Separation** - ETL, indexing, and search are cleanly separated  
✅ **Framework Flexibility** - Works with LlamaIndex, LangChain, or custom systems  
✅ **Production Ready** - Follows enterprise software design patterns  

## 🔮 Future Extensions Made Easy

This architecture makes these advanced features straightforward to add:

- **Vector Search** (embeddings, FAISS, Pinecone)
- **Hybrid Retrieval** (BM25 + vector combination)  
- **LLM-Enhanced Search** (query expansion, re-ranking)
- **Multi-Modal Search** (text, images, audio)
- **Graph-Based Retrieval** (knowledge graphs)
- **Real-Time Search** (streaming updates)

## 📊 Example Results

```bash
$ python app/demo_swappable_retrievers.py

==================================================
Testing BM25 Keyword Retriever
==================================================
Query: 'mathematician computer'
Found 1 results:
  1. Ada Lovelace (Score: 0.918)

==================================================  
Testing Mock Vector Similarity Retriever
==================================================
Query: 'mathematician computer'
Found 1 results:
  1. Ada Lovelace (Score: 1.000)

Demo Complete!
Notice how the client code worked with both retrievers without any changes!
This is the power of dependency inversion.
```

## Configuration

### Environment Variables

Create a `.env` file in the project root with your API credentials:

```env
# Azure AI Foundry (Recommended - Cloud)
AZURE_AI_ENDPOINT=https://your-endpoint.inference.ai.azure.com
AZURE_AI_API_KEY=your-api-key-here
AZURE_AI_MODEL=gpt-4o-mini

# Alternative Providers (Optional)
HUGGINGFACE_API_KEY=your-hf-token-here
OPENAI_API_KEY=your-openai-key-here
```

### LLM Priority Order

The system tries LLMs in this order:
1. **Azure AI Foundry** (cloud) - if credentials provided
2. **Ollama** (local) - if installed and running
3. **Hugging Face** (cloud) - if API key provided
4. **OpenAI** (cloud) - if API key provided
5. **No LLM** - retriever and tools still work!

### Azure AI Foundry Setup

1. Go to [Azure AI Foundry](https://ai.azure.com/)
2. Create or select a project
3. Get your endpoint URL and API key
4. Add them to your `.env` file
5. Choose a model (e.g., `gpt-4o-mini`, `gpt-4`, `gpt-3.5-turbo`)

## 🤝 Contributing

This project demonstrates architectural patterns rather than specific retrieval algorithms. Contributions that improve the **design patterns**, **documentation**, or **example implementations** are welcome!

## 📄 License

MIT License - Use this architecture in your projects!

---

**Built with ❤️ and SOLID principles. Perfect foundation for your next RAG system!** 🚀
