"""
Test Azure AI connect            llm = AzureOpenAI(
                engine=azure_model,  # deployment name (LlamaIndex uses 'engine')
                api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                api_version="2024-12-01-preview"
            )ecifically
"""
import asyncio
import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_azure_ai():
    try:
        from llama_index.llms.azure_openai import AzureOpenAI
        print("✅ Azure AI module imported successfully")

        azure_endpoint = os.getenv("AZURE_AI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_AI_API_KEY")
        azure_model = os.getenv("AZURE_AI_MODEL", "gpt-4o-mini")

        print(f"Endpoint: {azure_endpoint}")
        print(f"Model: {azure_model}")
        print(f"API Key: {'✅ Present' if azure_api_key else '❌ Missing'}")

        if azure_endpoint and azure_api_key:
            print("\n🔄 Testing Azure AI connection...")

            llm = AzureOpenAI(
                engine=azure_model,  # deployment name (LlamaIndex uses 'engine')
                api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                api_version="2024-12-01-preview"
            )

            # Test a simple completion
            response = await llm.acomplete("Hello, this is a test.")
            print("✅ Azure AI connection successful!")
            print(f"Response: {response.text[:100]}...")

        else:
            print("❌ Missing Azure AI configuration")

    except ImportError as e:
        print(f"❌ Azure AI module not available: {e}")
    except Exception as e:
        print(f"❌ Azure AI connection failed: {e}")
        print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    asyncio.run(test_azure_ai())
