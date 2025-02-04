from rag_system import RAGSystem

def test_rag():
    # Initialize RAG with debug mode
    rag = RAGSystem(debug=True)
    
    # Test question
    question = "what is terratracer"
    model = "mistral:latest"  # Using Mistral as it's generally good for this type of task
    
    print(f"\nTesting RAG system with question: {question}")
    print(f"Using model: {model}")
    
    answer = rag.answer_question(question, model)
    print(f"\nFinal answer: {answer}")

if __name__ == "__main__":
    test_rag() 