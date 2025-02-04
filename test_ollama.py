import requests
import json
import time

def test_ollama_api():
    """Test Ollama API directly"""
    base_url = "http://localhost:11434"
    
    # Test 1: Check if API is responding
    print("1. Testing API connection...")
    try:
        response = requests.get(f"{base_url}/api/tags")
        print(f"API Response Status: {response.status_code}")
        print(f"Available Models: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"API Connection Failed: {str(e)}")
    
    # Test 2: Simple generation
    print("\n2. Testing simple generation...")
    try:
        data = {
            "model": "qwen2.5",
            "prompt": "Say hello!",
            "stream": False
        }
        response = requests.post(f"{base_url}/api/generate", json=data)
        print(f"Generation Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Generation Failed: {str(e)}")

if __name__ == "__main__":
    test_ollama_api() 