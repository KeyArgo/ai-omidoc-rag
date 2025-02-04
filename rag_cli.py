from rag_system import RAGSystem
import argparse

def main():
    parser = argparse.ArgumentParser(description='RAG System CLI')
    parser.add_argument('--model', default='llama2', help='Ollama model to use')
    parser.add_argument('--db', default='./chroma_db', help='ChromaDB path')
    args = parser.parse_args()
    
    rag = RAGSystem(db_path=args.db)
    
    print(f"\n=== RAG System with {args.model} ===")
    print("Enter your questions (or 'quit' to exit)")
    
    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() in ['quit', 'exit']:
            break
            
        if query:
            answer = rag.answer_question(query, args.model)
            print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main() 