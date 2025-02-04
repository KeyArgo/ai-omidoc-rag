import requests

def check_ollama():
    print("\n=== Checking Ollama Setup ===")
    
    # 1. Check if Ollama is running
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            print("✓ Ollama is running")
            models = response.json()
            print("\nAvailable models:")
            for model in models.get('models', []):
                print(f"- {model['name']}")
        else:
            print("✗ Ollama returned unexpected status:", response.status_code)
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to Ollama - is it running?")
        print("\nTo start Ollama:")
        print("1. Open a terminal")
        print("2. Run: ollama serve")
    except Exception as e:
        print("✗ Error checking Ollama:", str(e))
    
    # 2. Print Ollama version
    try:
        response = requests.get('http://localhost:11434/api/version')
        if response.status_code == 200:
            version = response.json().get('version')
            print(f"\nOllama version: {version}")
    except:
        pass

if __name__ == "__main__":
    check_ollama() 