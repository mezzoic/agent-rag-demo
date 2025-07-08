import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_google_search():
    """Test Google Custom Search API directly"""

    # Get credentials
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    print(f"API Key: {api_key[:10]}...{api_key[-4:] if api_key else None}") # type: ignore
    print(f"Engine ID: {engine_id}")

    if not api_key or not engine_id:
        print("Missing credentials!")
        return

    # Test 1: Direct HTTP call
    import requests
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': api_key,
            'cx': engine_id,
            'q': 'test search'
        }

        print("\n=== Testing Direct HTTP Call ===")
        response = requests.get(url, params=params)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Found {len(data.get('items', []))} results")
            if 'items' in data:
                print(f"First result: {data['items'][0]['title']}")
        else:
            print(f"Error response: {response.text}")

    except Exception as e:
        print(f"HTTP test failed: {e}")

    # Test 2: LlamaIndex GoogleSearchToolSpec
    try:
        print("\n=== Testing LlamaIndex Tool ===")
        from llama_index.tools.google import GoogleSearchToolSpec

        search_tool = GoogleSearchToolSpec(key=api_key, engine=engine_id)
        result = search_tool.google_search("test search")

        print(f"LlamaIndex result type: {type(result)}")
        print(f"LlamaIndex result: {result}")

    except ImportError:
        print("LlamaIndex Google tools not installed. Run: uv add llama-index-tools-google")
    except Exception as e:
        print(f"LlamaIndex test failed: {e}")

if __name__ == "__main__":
    test_google_search()