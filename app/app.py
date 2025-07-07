import asyncio
import os

from dotenv import load_dotenv
from infrastructure.retriever import BaseRetriever, GuestRetriever, RetrieverInterface
from infrastructure.tools import (
    HuggingFaceModelSearchTool,
    QueryTool,
    WeatherInfoTool,
    WebSearchTool,
)
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context  # Add this import

# Load environment variables
load_dotenv()

# Try to import Ollama
try:
    from llama_index.llms.ollama import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    try:
        from llama_index_llms_ollama import Ollama  # type: ignore
        OLLAMA_AVAILABLE = True
    except ImportError:
        OLLAMA_AVAILABLE = False

# Try to import Hugging Face
try:
    from llama_index.llms.huggingface import HuggingFaceInferenceAPI  # type: ignore
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Try to import OpenAI (if available)
try:
    from llama_index.llms.openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import Azure AI
try:
    from llama_index.llms.azure_openai import AzureOpenAI
    AZURE_AI_AVAILABLE = True
except ImportError:
    AZURE_AI_AVAILABLE = False

async def main():
    # Initialize the retriever with complete ETL pipeline
    # Type annotation uses interface, but instantiation uses concrete class
    retriever: RetrieverInterface = GuestRetriever()
    guest_info_tool = QueryTool(retriever)
    weather_info_tool = WeatherInfoTool()
    hub_stats_tool = HuggingFaceModelSearchTool()
    # Initialize the web search tool
    # This requires GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID to be set in .env file

    google_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    google_search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    if not google_api_key or not google_search_engine_id:
        print("Web search tool requires GOOGLE_API_KEY and " \
        "GOOGLE_SEARCH_ENGINE_ID to be set in .env file"\
        f" {google_api_key},{ google_search_engine_id}")
        return
    try:
        search_tool = WebSearchTool(
            google_api_key=google_api_key,
            google_search_engine_id=google_search_engine_id
        ) # web search tool
        print("Web search tool initialized successfully")
    except Exception as e:
        print(f"Failed to initialize web search tool: {e}")
        return

    # Set up LLM with fallbacks:
    # Azure AI (cloud) -> Ollama (local) -> Hugging Face (cloud) -> OpenAI (cloud)
    demoWorkflow = None
    llm = None

    # Try Azure AI first (cloud)
    if AZURE_AI_AVAILABLE:
        azure_endpoint = os.getenv("AZURE_AI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_AI_API_KEY")
        azure_model = os.getenv("AZURE_AI_MODEL", "gpt-4o-mini")

        if azure_endpoint and azure_api_key:
            try:
                llm = AzureOpenAI(  # type: ignore
                    engine=azure_model,  # deployment name (LlamaIndex uses 'engine')
                    api_key=azure_api_key,
                    azure_endpoint=azure_endpoint,
                    api_version="2024-12-01-preview"
                )
                demoWorkflow = AgentWorkflow.from_tools_or_functions(
                    [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool],
                    llm=llm
                )
                print(f"LLM workflow enabled with Azure AI ({azure_model})")
            except Exception as e:
                print(f"Azure AI setup failed: {e}")
                llm = None
        else:
            print(
                "Azure AI available but missing configuration "
                "(set AZURE_AI_ENDPOINT and AZURE_AI_API_KEY)"
            )

    # Try Ollama if Azure AI failed (local)
    if not llm and OLLAMA_AVAILABLE:
        try:
            # Try your installed models (from ollama list)
            # Note: Some models may not support tools/function calling
            for model in ["llama3.1:8b", "qwen2.5-coder:1.5b-base", "gemma3:1b"]:
                try:
                    llm = Ollama(model=model, request_timeout=60.0)  # type: ignore
                    print(f"Using Ollama model: {model}")
                    break
                except Exception as e:
                    print(f"Model {model} not available: {e}")
                    continue

            if llm:
                # Create workflow
                demoWorkflow = AgentWorkflow.from_tools_or_functions(
                    [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool],
                    llm=llm
                )
                print("LLM workflow enabled with Ollama")
        except Exception as e:
            print(f"Ollama setup failed: {e}")
            llm = None

    # Try Hugging Face if Ollama failed
    if not llm and HUGGINGFACE_AVAILABLE:
        hf_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
        if hf_token:
            try:
                llm = HuggingFaceInferenceAPI(  # type: ignore
                    model_name="microsoft/DialoGPT-medium",
                    token=hf_token
                )
                demoWorkflow = AgentWorkflow.from_tools_or_functions(
                    [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool],
                    llm=llm
                )
                print("LLM workflow enabled with Hugging Face")
            except Exception as e:
                print(f"Hugging Face setup failed: {e}")
                llm = None
        else:
            print(
                "Hugging Face available but no API token found "
                "(set HUGGINGFACE_API_KEY or HF_TOKEN)"
            )

    # Try OpenAI if both Ollama and HF failed
    if not llm and OPENAI_AVAILABLE:
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                llm = OpenAI(model="gpt-3.5-turbo", api_key=openai_key)  # type: ignore
                demoWorkflow = AgentWorkflow.from_tools_or_functions(
                    [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool],
                    llm=llm
                )
                print("LLM workflow enabled with OpenAI")
            except Exception as e:
                print(f"OpenAI setup failed: {e}")
                llm = None
        else:
            print("OpenAI available but no API key found (set OPENAI_API_KEY)")

    # Final fallback message
    if not llm:
        print("\n" + "="*60)
        print("No LLM available. To enable LLM workflows:")
        print("1. Azure AI Foundry (recommended):")
        print("   - Set AZURE_AI_ENDPOINT in .env file")
        print("   - Set AZURE_AI_API_KEY in .env file")
        print("   - Optionally set AZURE_AI_MODEL (default: gpt-4o-mini)")
        print("2. Install Ollama locally: https://ollama.ai/")
        print("   Then run: ollama pull llama3.1:8b")
        print("3. Or set HUGGINGFACE_API_KEY for cloud inference")
        print("4. Or set OPENAI_API_KEY for OpenAI")
        print("The retriever and tools will still work without LLM!")
        print("="*60)

    # Test different approaches

    if demoWorkflow:
        print("\n=== Using Workflow (LLM + Tool) ===")
        ctx = Context(demoWorkflow)
        try:
            queries = [
                "Tell me about Lady Ada Lovelace. What's her background?",
                "What's the weather like in Paris tonight? Will it be suitable for our fireworks display?",
                "One of our guests is from Google. What can you tell me about their most popular model?",
                "I need to speak with Dr. Nikola Tesla about recent advancements in wireless energy. Can you help me prepare for this conversation?"
            ]
            for query in queries:
                print(f"\nQuery: {query}")

                workflow_response = await demoWorkflow.run(query, ctx=ctx)
                print("Workflow Response:")
                print(workflow_response)

        except Exception as e:
            print(f"Workflow error: {e}")
            print("Continuing with other tests...")
    query = "mathematician"
    print("\n=== Using Tool Directly ===")
    tool_result = guest_info_tool(query)
    print("Tool Result Content:")
    print(tool_result.content)

    print("\n=== Using Raw Retriever ===")
    raw_results = retriever.search(query)
    print(f"Found {len(raw_results)} raw results:")
    for i, result in enumerate(raw_results, 1):
        score = result.score if hasattr(result, 'score') else 'N/A'
        name = result.node.metadata.get("name", f"Guest {i}")
        print(f"{i}. {name} (Score: {score})")

    print("\nFormatted results:")
    formatted_detailed = BaseRetriever.format_NodeWithScores(raw_results, "detailed")
    print(formatted_detailed[:500] + "..." if len(formatted_detailed) > 500 else formatted_detailed)




if __name__ == "__main__":
    asyncio.run(main())

