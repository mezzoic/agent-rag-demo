# Cloud-Only Setup Example

This example shows how to use the RAG system with only cloud providers (no local Ollama required).

## Quick Start with Azure AI Foundry

1. **Get Azure AI credentials:**
   - Go to [Azure AI Foundry](https://ai.azure.com/)
   - Create a project and get your endpoint + API key

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your Azure AI credentials
   ```

3. **Run the demo:**
   ```bash
   python app/app.py
   ```

## Example `.env` for Cloud-Only Setup

```env
# Azure AI Foundry (Primary)
AZURE_AI_ENDPOINT=https://myproject.inference.ai.azure.com
AZURE_AI_API_KEY=sk-abcd1234...
AZURE_AI_MODEL=gpt-4o-mini

# Fallback options (optional)
HUGGINGFACE_API_KEY=hf_abcd1234...
OPENAI_API_KEY=sk-abcd1234...
```

## Benefits of Cloud Setup

- ✅ **No local installation** - works immediately
- ✅ **Latest models** - access to GPT-4o, Claude, etc.
- ✅ **Scalable** - no GPU/memory constraints
- ✅ **Reliable** - enterprise-grade infrastructure
- ✅ **Cost-effective** - pay per use

## Cost Considerations

- **Azure AI Foundry**: ~$0.15-0.30 per 1K tokens (GPT-4o-mini)
- **Hugging Face**: Free tier available, then ~$0.50/hour
- **OpenAI**: ~$0.15-0.30 per 1K tokens

For development/testing, the costs are typically under $1-5 per month.
