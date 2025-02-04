from rag_system import RAGSystem
import json

def test_vector_search():
    """Test the vector search functionality of the RAG system"""
    print("\n=== Testing Vector Search ===")
    
    # Initialize RAG system
    rag = RAGSystem(debug=True)
    
    # Test queries
    test_queries = [
        "what is terratracer",
        "how to use terratracer",
        "terratracer features",
        "RAG system",
        "LLM integration"
    ]
    
    for query in test_queries:
        print(f"\n\n--- Testing Query: {query} ---")
        
        # Get documents without LLM processing
        docs = rag.query_knowledge(query)
        
        print(f"\nFound {len(docs)} documents")
        print("\nDocument Previews:")
        
        for i, doc in enumerate(docs, 1):
            # Get first 200 chars of each document
            preview = doc[:200] + "..." if len(doc) > 200 else doc
            print(f"\nDocument {i}:")
            print("-" * 80)
            print(preview)
            print("-" * 80)
            
            # Try to parse JSON if present
            try:
                json_data = json.loads(doc)
                if isinstance(json_data, dict):
                    print("\nDocument Metadata:")
                    for key, value in json_data.items():
                        print(f"- {key}: {value}")
            except:
                pass

def test_specific_search():
    """Test search with specific terms we know should be in the database"""
    print("\n=== Testing Specific Terms ===")
    
    rag = RAGSystem(debug=False)  # Disable debug for cleaner output
    
    # Add terms we know should be in the database
    known_terms = [
        "TerraTracer_v0.1.1.py",
        "display_monument_point",
        "KML creation",
        "coordinate_format"
    ]
    
    for term in known_terms:
        print(f"\n--- Searching for: {term} ---")
        docs = rag.query_knowledge(term)
        
        if docs:
            print(f"Found {len(docs)} relevant documents")
            # Show brief preview of first document
            preview = docs[0][:150] + "..." if len(docs[0]) > 150 else docs[0]
            print("\nFirst document preview:")
            print(preview)
        else:
            print("No documents found")

if __name__ == "__main__":
    print("Testing RAG vector search functionality...")
    
    # Test general vector search
    test_vector_search()
    
    print("\n" + "="*80 + "\n")
    
    # Test specific known terms
    test_specific_search() 