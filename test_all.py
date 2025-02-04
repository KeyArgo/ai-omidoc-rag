from rag_system import RAGSystem
import json
import argparse

def test_vector_search(query: str, num_results: int = 3, timeout: int = 300):
    """Test vector search with custom query"""
    print(f"\n=== Testing Vector Search for: '{query}' ===")
    
    rag = RAGSystem(debug=True, timeout=timeout)
    docs = rag.query_knowledge(query, max_results=num_results)
    
    print(f"\nFound {len(docs)} documents")
    
    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i}:")
        print("=" * 80)
        
        # Try to parse as JSON first
        try:
            data = json.loads(doc)
            if isinstance(data, dict):
                print("JSON Content:")
                print(json.dumps(data, indent=2))
            else:
                print(doc[:500] + "..." if len(doc) > 500 else doc)
        except:
            # Not JSON, show as text
            print(doc[:500] + "..." if len(doc) > 500 else doc)
        
        print("=" * 80)

def test_rag_query(query: str, model: str = "mistral:latest", timeout: int = 300):
    """Test full RAG pipeline with custom query and model"""
    print(f"\n=== Testing RAG Pipeline ===")
    print(f"Query: {query}")
    print(f"Model: {model}")
    
    rag = RAGSystem(debug=True, timeout=timeout)
    answer = rag.answer_question(query, model)
    
    print("\nFinal Answer:")
    print("-" * 40)
    print(answer)
    print("-" * 40)

def get_valid_query() -> str:
    """Get a valid query from user input"""
    while True:
        query = input("\nEnter search query (or 'quit' to exit): ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            return None
        if len(query) < 3:
            print("Please enter a longer query (at least 3 characters)")
            continue
        return query

def get_valid_number(prompt: str, default: int = 3) -> int:
    """Get a valid number from user input"""
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input:
                return default
            num = int(user_input)
            if num > 0:
                return num
            print("Please enter a positive number")
        except ValueError:
            print(f"Invalid input. Using default value: {default}")
            return default

def main():
    parser = argparse.ArgumentParser(description='Test RAG System')
    parser.add_argument('--mode', choices=['vector', 'rag', 'both'], 
                      default='both', help='Test mode')
    parser.add_argument('--timeout', type=int, default=300,
                      help='Timeout in seconds (default: 300)')
    args = parser.parse_args()
    
    print("\n=== RAG System Testing Tool ===")
    print(f"\nTimeout set to: {args.timeout} seconds")
    print("\nAvailable models:")
    print("1. mistral:latest")
    print("2. gemma:2b (faster)")
    print("3. phi4:latest (faster)")
    print("4. qwen2.5:14b")
    print("5. mixtral:latest")
    
    while True:
        query = get_valid_query()
        if query is None:
            break
            
        if args.mode in ['vector', 'both']:
            num_results = get_valid_number("Number of results to show (default 3): ")
            test_vector_search(query, num_results, args.timeout)
        
        if args.mode in ['rag', 'both']:
            model = input("Enter model name (default: mistral:latest): ").strip() or "mistral:latest"
            test_rag_query(query, model, args.timeout)
        
        print("\n" + "="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}") 