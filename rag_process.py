from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
import chromadb
import json
import os
from tqdm.auto import tqdm
import time
import torch
import sys
import pickle
from datetime import datetime
import glob

def monitor_gpu():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # Convert to GB
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        print(f"Free Memory: {12 - reserved:.2f}GB")  # 12GB total for 4070Ti

def check_system_requirements():
    """Check if system meets requirements"""
    requirements = {
        "gpu": False,
        "cuda": False,
        "vram": 0,
        "pytorch": False
    }
    
    try:
        import torch
        requirements["pytorch"] = True
        
        if torch.cuda.is_available():
            requirements["gpu"] = True
            requirements["cuda"] = True
            requirements["vram"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            
            print("\nSystem Check:")
            print(f"✓ GPU: {torch.cuda.get_device_name()}")
            print(f"✓ CUDA: {torch.version.cuda}")
            print(f"✓ VRAM: {requirements['vram']:.1f}GB")
        else:
            print("\nNo NVIDIA GPU detected, will use CPU (slower)")
    except Exception as e:
        print(f"\nError checking system: {str(e)}")
    
    return requirements

def setup_gpu():
    """Configure GPU settings based on available hardware"""
    reqs = check_system_requirements()
    
    if reqs["gpu"] and reqs["cuda"]:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
            # Adjust settings based on available VRAM
            vram = reqs["vram"]
            if vram >= 12:  # High-end GPUs (4070Ti, 3080, etc)
                return "high"
            elif vram >= 8:  # Mid-range GPUs (3070, etc)
                return "medium"
            elif vram >= 4:  # Entry GPUs (3060, etc)
                return "low"
            else:
                print("⚠ Warning: Low VRAM detected, using conservative settings")
                return "minimal"
        except Exception as e:
            print(f"GPU initialization failed: {str(e)}")
            return None
    return None

def setup_ollama_phi():
    """Initialize Phi-4 through Ollama"""
    llm = Ollama(
        model="phi4",
        temperature=0.7,
        context_window=4096,  # Back to full context
        additional_kwargs={
            "top_p": 0.95,
            "num_ctx": 4096,
            "repeat_penalty": 1.1,
            "num_gpu": 1,
            "num_thread": 8
        }
    )
    return llm

def process_json_in_chunks(file_path, chunk_size=50):  # Increased for better performance
    print(f"\n1. Reading JSON file: {file_path}")
    start_time = time.time()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        print("2. Loading JSON into memory...")
        data = json.load(f)
        print(f"   ✓ Loaded {len(data):,} entries in {time.time() - start_time:.2f} seconds")
    
    # Remove limit for full processing
    total_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size else 0)
    print(f"\n3. Will process {len(data):,} entries in {total_chunks:,} chunks")
    print(f"   Each chunk contains {chunk_size} entries")
    
    documents = []
    
    for i in tqdm(range(0, len(data), chunk_size), desc="Processing chunks"):
        chunk = data[i:i + chunk_size]
        chunk_text = json.dumps(chunk, indent=2)
        documents.append(Document(text=chunk_text, doc_id=f"chunk_{i}"))
        
        if len(documents) >= 10 or i + chunk_size >= len(data):  # Increased batch size
            yield documents
            documents = []

def save_checkpoint(checkpoint_data, persist_dir):
    """Save processing checkpoint"""
    checkpoint_file = f"{persist_dir}/checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"\n📑 Checkpoint saved: {checkpoint_file}")

def load_latest_checkpoint(persist_dir):
    """Load most recent checkpoint if exists"""
    try:
        checkpoints = sorted(glob.glob(f"{persist_dir}/checkpoint_*.pkl"))
        if checkpoints:
            with open(checkpoints[-1], 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"No valid checkpoint found: {e}")
    return None

def create_rag_index(data_dir="./data", persist_dir="./chroma_db"):
    print("\n=== Starting RAG Index Creation ===")
    
    # Check system capabilities
    gpu_tier = setup_gpu()
    
    # Adjust batch sizes based on GPU tier
    if gpu_tier == "high":
        embed_batch_size = 64
        chunk_size = 50
    elif gpu_tier == "medium":
        embed_batch_size = 32
        chunk_size = 30
    elif gpu_tier == "low":
        embed_batch_size = 16
        chunk_size = 20
    else:  # CPU or minimal GPU
        embed_batch_size = 8
        chunk_size = 10
    
    print("\n1. Initializing Phi-4...")
    llm = setup_ollama_phi()
    print("   ✓ Phi-4 initialized through Ollama")
    
    json_file = os.path.join(data_dir, "ChatGPT Conversations.json")
    if not os.path.exists(json_file):
        print(f"Error: JSON file not found at {json_file}")
        return None
    
    print("\n2. Setting up ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    
    try:
        chroma_client.delete_collection("documents")
    except:
        pass
    
    chroma_collection = chroma_client.create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 100}
    )
    
    print("\n3. Initializing vector store and embeddings...")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Adjust embedding model settings based on hardware
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        embed_batch_size=embed_batch_size,
        device="cuda" if gpu_tier else "cpu",
        model_kwargs={
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if gpu_tier else torch.float32,
            "device_map": "auto" if gpu_tier else None
        }
    )
    
    # Set up global settings
    Settings.embed_model = embed_model
    Settings.llm = llm  # Set Phi-4 as the default LLM
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Optimize node parser settings
    node_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    
    # Calculate total expected nodes
    total_entries = len(json.load(open(json_file, 'r', encoding='utf-8')))
    estimated_total_nodes = total_entries * 1.3
    
    # Initialize progress bar
    pbar = tqdm(total=int(estimated_total_nodes), 
                desc="Processing nodes", 
                unit="nodes",
                file=sys.stdout)
    
    print(f"\nEstimated total nodes to process: {int(estimated_total_nodes):,}")
    
    print("\n4. Starting document processing...")
    stats = {
        'total_nodes': 0,
        'total_chunks': 0,
        'start_time': time.time(),
        'last_update': time.time(),
        'batch_times': [],
        'nodes_per_batch': [],
        'embedding_times': [],
        'checkpoint_interval': 1000  # Save checkpoint every 1000 nodes
    }
    
    try:
        for doc_batch in process_json_in_chunks(json_file):
            batch_start = time.time()
            
            # Create nodes
            nodes = node_parser.get_nodes_from_documents(doc_batch)
            
            # Time embedding generation
            embed_start = time.time()
            batch_index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                embed_model=embed_model,
                llm=llm,
                show_progress=False  # We're using our own progress bar
            )
            embed_time = time.time() - embed_start
            
            # Update statistics
            batch_time = time.time() - batch_start
            stats['total_nodes'] += len(nodes)
            stats['total_chunks'] += len(doc_batch)
            stats['batch_times'].append(batch_time)
            stats['nodes_per_batch'].append(len(nodes))
            stats['embedding_times'].append(embed_time)
            
            # Update progress bar
            pbar.update(len(nodes))
            
            # Save checkpoint if needed
            if stats['total_nodes'] % stats['checkpoint_interval'] == 0:
                checkpoint_data = {
                    'stats': stats,
                    'storage_context': storage_context,
                    'processed_chunks': stats['total_chunks']
                }
                save_checkpoint(checkpoint_data, persist_dir)
            
            current_time = time.time()
            if current_time - stats['last_update'] >= 5:
                elapsed_time = current_time - stats['start_time']
                nodes_per_second = stats['total_nodes'] / elapsed_time
                remaining_nodes = estimated_total_nodes - stats['total_nodes']
                eta = remaining_nodes / nodes_per_second if nodes_per_second > 0 else 0
                
                print(f"\n📊 Detailed Progress Report:")
                print(f"├── Nodes: {stats['total_nodes']:,} / {int(estimated_total_nodes):,} ({(stats['total_nodes']/estimated_total_nodes)*100:.1f}%)")
                print(f"├── Chunks: {stats['total_chunks']:,} / {778:,}")
                print(f"├── Speed: {nodes_per_second:.1f} nodes/second")
                print(f"├── Average batch size: {sum(stats['nodes_per_batch'])/len(stats['nodes_per_batch']):.1f} nodes")
                print(f"├── Average embedding time: {sum(stats['embedding_times'])/len(stats['embedding_times']):.2f}s")
                print(f"├── Time elapsed: {elapsed_time/60:.1f} minutes")
                print(f"└── ETA: {eta/60:.1f} minutes")
                
                stats['last_update'] = current_time
    
    except Exception as e:
        print(f"\n⚠ Error during processing: {str(e)}")
        # Save emergency checkpoint
        save_checkpoint({'stats': stats, 'error': str(e)}, persist_dir)
        return None
    finally:
        pbar.close()
    
    print(f"\n✅ Processing completed!")
    print(f"\n📈 Final Statistics:")
    print(f"├── Total nodes processed: {stats['total_nodes']:,}")
    print(f"├── Total chunks processed: {stats['total_chunks']:,}")
    print(f"├── Total time: {(time.time() - stats['start_time'])/60:.1f} minutes")
    print(f"├── Average speed: {stats['total_nodes']/(time.time() - stats['start_time']):.1f} nodes/second")
    print(f"├── Average batch time: {sum(stats['batch_times'])/len(stats['batch_times']):.2f}s")
    print(f"└── Average embedding time: {sum(stats['embedding_times'])/len(stats['embedding_times']):.2f}s")
    
    return batch_index

if __name__ == "__main__":
    print("=== RAG Processing Started ===")
    index = create_rag_index()
