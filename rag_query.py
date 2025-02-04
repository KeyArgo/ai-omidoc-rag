from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


def load_rag_index(persist_dir="./chroma_db"):
    # Reconnect to ChromaDB
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_collection("documents")
    
    # Initialize vector store
    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )
    
    # Create storage context
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    
    # Load index
    index = load_index_from_storage(storage_context)
    return index

def query_documents(query: str, index):
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return response

if __name__ == "__main__":
    index = load_rag_index()
    
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        response = query_documents(query, index)
        print(f"\nAnswer: {response}")
