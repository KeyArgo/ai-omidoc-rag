from llama_index.core import VectorStoreIndex, StorageContext, Settings, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb
import time
import subprocess
import json
import torch
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np

# First, let's try to install any missing packages
try:
    import pkg_resources
    pkg_resources.require(['llama-index-vector-stores-chroma'])
except:
    import subprocess
    print("Installing missing packages...")
    subprocess.check_call(['pip', 'install', 'llama-index-vector-stores-chroma'])

# Global cache for embeddings model
EMBEDDING_MODEL = None

def get_embedding_model():
    """Get or initialize embedding model"""
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        print("Loading embedding model (one-time operation)...")
        EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    return EMBEDDING_MODEL

def get_ollama_models():
    """Get list of installed Ollama models"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        models = []
        for line in result.stdout.split('\n')[1:]:  # Skip header
            if line.strip():
                model_name = line.split()[0]
                models.append(model_name)
        return sorted(models)
    except Exception as e:
        print(f"Error getting models: {e}")
        return []

def test_ollama_connection(model_name):
    """Test if Ollama model is responding using direct command"""
    print(f"\nTesting connection to {model_name}...")
    print("(First load may take several minutes)")
    
    try:
        # First ensure no other models are running
        subprocess.run(['ollama', 'stop'], capture_output=True)
        time.sleep(5)  # Wait for cleanup
        
        print("Sending test message via command line...")
        # Use different encoding and error handling
        process = subprocess.Popen(
            ['ollama', 'run', model_name, 'hi'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='ignore'
        )
        
        # Wait for response with timeout
        try:
            stdout, stderr = process.communicate(timeout=600)  # 10 minutes timeout
            if process.returncode == 0:
                print(f"Test successful! Response: {stdout.strip()}")
                return True
            else:
                print(f"Test failed with error: {stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            process.kill()
            print("Test timed out after 10 minutes")
            return False
            
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        return False

def setup_rag_query(model_name):
    print(f"1. Initializing ChromaDB...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_collection("documents")
    
    print("2. Setting up vector store with size limits...")
    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection,
        store_kwargs={
            "limit": 5  # Only retrieve top 5 most relevant chunks
        }
    )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print("3. Loading embeddings model...")
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        device="cpu",
        embed_batch_size=1
    )
    
    print("4. Creating index...")
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
        show_progress=True,
        use_async=False  # Force synchronous for stability
    )
    
    print(f"5. Setting up {model_name}...")
    llm = Ollama(
        model=model_name,
        temperature=0.1,
        additional_kwargs={
            "num_gpu": 1,
            "num_thread": 4,
            "timeout": 30,  # Even shorter timeout
            "context_window": 512,  # Minimal context
            "num_ctx": 512,
            "num_predict": 64,  # Very short responses
            "stop": ["</response>", "Human:", "Assistant:"],
            "repeat_penalty": 1.1,
            "top_k": 5,
            "top_p": 0.5,
            "seed": 42,
            "mirostat": 2,  # Add adaptive sampling
            "mirostat_tau": 5.0,
            "mirostat_eta": 0.1
        }
    )
    
    print("6. Creating query engine...")
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=1,
        response_mode="compact",  # Most minimal response mode
        streaming=False,
        context_window=512,
        num_output=64,
        node_postprocessors=[],
        verbose=True,
        similarity_cutoff=0.8,  # More strict relevance
        max_tokens=64,
        max_chunks=5  # Limit chunks processed
    )
    
    return query_engine

def query_with_progress(query_engine, question):
    """Execute query with progress updates"""
    print("\nProcessing query...")
    start_time = time.time()
    last_update = start_time
    update_interval = 5  # Update every 5 seconds
    
    try:
        # Start a thread to show progress
        def show_progress():
            nonlocal last_update
            while True:
                current_time = time.time()
                if current_time - last_update >= update_interval:
                    elapsed = current_time - start_time
                    print(f"Still processing... ({elapsed:.0f}s elapsed)")
                    last_update = current_time
                time.sleep(1)
        
        import threading
        progress_thread = threading.Thread(target=show_progress, daemon=True)
        progress_thread.start()
        
        # Execute query with timeout
        response = query_engine.query(question)
        
        # Query completed
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n✓ Query completed in {total_time:.1f} seconds")
        
        # Save timing information
        with open("query_times.txt", "a") as f:
            f.write(f"{datetime.now()}, {question}, {total_time:.1f}s\n")
        
        return response
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n❌ Query failed after {total_time:.1f} seconds: {str(e)}")
        with open("query_times.txt", "a") as f:
            f.write(f"{datetime.now()}, {question}, FAILED after {total_time:.1f}s: {str(e)}\n")
        raise

def check_documents():
    """Check if documents are loaded in ChromaDB"""
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_collection("documents")
        count = collection.count()
        print(f"\nFound {count} documents in ChromaDB")
        if count > 0:
            # Get a sample to verify content
            sample = collection.peek()
            print(f"Collection contains data about ChatGPT conversations")
            return True
    except Exception as e:
        print(f"Error checking ChromaDB: {str(e)}")
    return False

def load_documents():
    """Check if documents are already processed"""
    print("\nChecking ChromaDB status...")
    
    if check_documents():
        print("✓ Documents already processed and loaded in ChromaDB!")
        return True
        
    print("\n❌ No processed documents found in ChromaDB.")
    print("Please run rag_process.py first to process the ChatGPT Conversations.json file.")
    return False

def direct_query(model_name, question, max_results=5):
    """Simple document search optimized for model use"""
    print(f"\nSearching for '{question}'...")
    start_time = time.time()
    
    try:
        # 1. Setup ChromaDB
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_collection("documents")
        
        # 2. Simple search
        results = collection.query(
            query_texts=[question],
            n_results=max_results
        )
        
        # 3. Process results
        if not results or 'documents' not in results:
            return "No documents found"
            
        docs = results['documents'][0]
        distances = results.get('distances', [[]])[0]
        
        # 4. Format for model consumption
        response = []
        for i, (doc, dist) in enumerate(zip(docs, distances), 1):
            relevance = 1 - (dist / 2)
            response.append(f"Document {i} (Relevance: {relevance:.1%}):")
            
            # Clean and format content
            content = str(doc).strip()
            if content:
                response.append(content)
            response.append("---")
        
        return "\n".join(response)
        
    except Exception as e:
        return f"Search failed: {str(e)}"

def verify_chroma_data():
    """Verify ChromaDB data integrity"""
    try:
        print("\nVerifying ChromaDB data...")
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_collection("documents")
        
        # Get a sample of the data
        sample = collection.peek(10)  # Get 10 random documents
        
        print(f"\nFound {collection.count()} total documents")
        print("\nSample document preview:")
        
        for i, doc in enumerate(sample['documents'][:3], 1):
            # Safely truncate document for display
            preview = doc[:100] + "..." if len(doc) > 100 else doc
            print(f"\n{i}. {preview}")
            
        return True
        
    except Exception as e:
        print(f"\n❌ Error verifying data: {str(e)}")
        return False

def main():
    print("\n=== Chat History Search ===")
    print("\nThis tool searches through your chat history")
    print("\nTip: Try searching for specific terms like:")
    print("1. RAG")
    print("2. LLM")
    print("3. ChatGPT")
    print("\nOr try exact phrases in quotes like:")
    print('1. "what is rag"')
    print('2. "how to use llm"')
    print('3. "example of chatgpt"')
    
    # Verify data
    print("\nVerifying ChromaDB data...")
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_collection("documents")
        count = collection.count()
        print(f"Found {count} documents")
        
        # Show sample
        sample = collection.peek(5)
        print("\nSample documents:")
        for i, doc in enumerate(sample['documents'][:3], 1):
            print(f"\n{i}. {str(doc)[:200]}...")
            
    except Exception as e:
        print(f"Error verifying data: {e}")
        return
    
    # Search loop
    print("\nEnter search terms (or 'quit' to exit)")
    while True:
        query = input("\nSearch: ").strip()
        if query.lower() == 'quit':
            break
        if not query:
            continue
            
        result = direct_query("none", query)
        print(result)

if __name__ == "__main__":
    main()
