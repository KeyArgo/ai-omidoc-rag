from typing import List, Dict, Any
import chromadb
import json
import requests
import re
import time

class RAGSystem:
    def __init__(self, db_path: str = "./chroma_db", debug: bool = True, 
                 timeout: int = 300, min_relevance: float = 0.5):
        """Initialize RAG system with ChromaDB"""
        self.debug = debug
        self.timeout = timeout
        self.min_relevance = min_relevance
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_collection("documents")
        if self.debug:
            print(f"RAG System initialized with DB: {db_path}")
            print(f"Timeout set to: {timeout} seconds")
    
    def clean_text(self, text: str) -> str:
        """Clean and format text with better filtering"""
        # Skip error logs and stack traces
        if any(skip in text.lower() for skip in ['error:', 'traceback', 'exception:']):
            return ""
        
        # Skip git commit messages
        if 'commit' in text.lower() and any(git in text.lower() for git in ['branch', 'merge', 'push']):
            return ""
        
        # Remove file paths and timestamps
        text = re.sub(r'File ".*?"', '', text)
        text = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}', '', text)
        
        # Clean up JSON/dict formatting
        text = re.sub(r'["\'](\w+)["\']:', r'\1:', text)
        text = re.sub(r'author_role:|create_time:|content:', '', text)
        
        # Remove markdown and code formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def query_knowledge(self, query: str, max_results: int = 10) -> List[str]:
        """Get relevant documents with better search terms"""
        if self.debug:
            print(f"\nQuerying knowledge base for: {query}")
        
        if len(query.strip()) < 3:
            if self.debug:
                print("Query too short, skipping search")
            return []
        
        try:
            # Define search variations
            search_terms = [
                query,  # exact match
                f"description of {query}",
                f"{query} is a program",
                f"{query} features",
                f"{query} functionality",
                f"{query} purpose",
                f"{query} usage",
                f"how to use {query}",
                f"{query} documentation"
            ]
            
            all_docs = set()
            
            # Search with each term
            for term in search_terms:
                results = self.collection.query(
                    query_texts=[term],
                    n_results=max_results
                )
                
                if results and 'documents' in results:
                    for doc in results['documents'][0]:
                        cleaned = self.clean_text(doc)
                        if cleaned:  # Only add non-empty cleaned docs
                            all_docs.add(cleaned)
            
            # Sort documents by relevance
            scored_docs = []
            for doc in all_docs:
                score = 0
                # Increase score for important keywords
                keywords = ['terratracer', 'geospatial', 'lode', 'claim', 'land', 'certificate']
                for keyword in keywords:
                    if keyword in doc.lower():
                        score += 1
                # Increase score for longer, meaningful content
                if len(doc.split()) > 20:  # Prefer longer descriptions
                    score += 1
                scored_docs.append((score, doc))
            
            # Sort by score and return top results
            sorted_docs = [doc for score, doc in sorted(scored_docs, reverse=True)]
            
            if self.debug:
                print(f"Found {len(sorted_docs)} relevant documents")
            
            return sorted_docs[:max_results]
            
        except Exception as e:
            if self.debug:
                print(f"Error during query: {str(e)}")
            return []
    
    def format_context(self, documents: List[str], max_chars: int = 8000) -> str:
        """Format documents into context with better organization"""
        if self.debug:
            print("\nFormatting context...")
            
        formatted_docs = []
        total_chars = 0
        
        for i, doc in enumerate(documents, 1):
            # Extract key information
            key_info = doc[:200]  # Get first 200 chars for summary
            
            # Format document with summary
            doc_text = f"[Document {i}]\nSummary: {key_info}...\n\nFull content:\n{doc}\n"
            
            if total_chars + len(doc_text) > max_chars:
                break
            
            formatted_docs.append(doc_text)
            total_chars += len(doc_text)
        
        formatted = "\n---\n".join(formatted_docs)
        
        if self.debug:
            print(f"Context length: {len(formatted)} characters")
        return formatted
    
    def generate_prompt(self, query: str, context: str) -> str:
        """Create prompt with context and query"""
        if self.debug:
            print("\nGenerating prompt...")
            
        # Skip prompting for very short queries
        if len(query.strip()) < 3:
            return "Please provide a more specific query (at least 3 characters)."
        
        prompt = f"""Based on the following context, please answer the question. If the context doesn't contain relevant information, say so.
If the question is too vague or short, ask for clarification.

Context:
{context}

Question: {query}

Answer:"""

        if self.debug:
            print(f"Prompt length: {len(prompt)} characters")
        return prompt
    
    def query_ollama(self, model_name: str, prompt: str) -> str:
        """Query Ollama model with configurable timeout and streaming"""
        if self.debug:
            print(f"\nQuerying Ollama model: {model_name}")
            start_time = time.time()
            
        try:
            if self.debug:
                print("Sending request to Ollama...")
                
            # Health check with short timeout
            try:
                health_check = requests.get('http://localhost:11434/api/version', timeout=5)
                if health_check.status_code != 200:
                    return "Error: Ollama service is not responding correctly"
            except:
                return "Error: Could not connect to Ollama. Please make sure it's running (ollama serve)"
                
            # Model-specific parameters
            model_params = {
                'temperature': 0.7,
                'num_predict': 500,
                'top_k': 40,
                'top_p': 0.9
            }
            
            # Adjust parameters for faster models
            if any(fast_model in model_name.lower() for fast_model in ['gemma', 'phi', 'neural-chat']):
                model_params.update({
                    'temperature': 0.8,
                    'num_predict': 1000,  # Allow longer responses for faster models
                })
            
            # Use streaming for better performance
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': model_name,
                    'prompt': prompt,
                    'stream': True,
                    'options': model_params
                },
                stream=True,
                timeout=self.timeout  # Use configured timeout
            )
            
            if response.status_code != 200:
                return f"Error: Ollama returned status code {response.status_code}"
            
            # Stream the response
            full_response = []
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line)
                        if 'response' in json_response:
                            if self.debug:
                                print(json_response['response'], end='', flush=True)
                            full_response.append(json_response['response'])
                    except json.JSONDecodeError:
                        continue
                        
            if self.debug:
                print()  # New line after streaming
                elapsed_time = time.time() - start_time
                print(f"\nResponse completed in {elapsed_time:.2f} seconds")
                
            return ''.join(full_response)
                
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama. Is it running?"
        except requests.exceptions.Timeout:
            return f"Error: Request to Ollama timed out after {self.timeout} seconds. Try reducing context length or using a faster model."
        except Exception as e:
            if self.debug:
                print(f"Error details: {str(e)}")
            return f"Error querying Ollama: {str(e)}"
    
    def answer_question(self, query: str, model_name: str = "mistral:latest") -> str:
        """Complete RAG pipeline with model-specific optimizations"""
        if self.debug:
            print(f"\n=== Processing question: {query} ===")
            print(f"Using model: {model_name}")
            
        # 1. Get relevant documents
        documents = self.query_knowledge(query)
        if not documents:
            return "No relevant information found."
            
        # 2. Format context with model-specific length
        max_chars = 8000  # Default context length
        if any(fast_model in model_name.lower() for fast_model in ['gemma', 'phi']):
            max_chars = 12000  # Allow longer context for faster models
            
        context = self.format_context(documents, max_chars=max_chars)
        
        # 3. Generate prompt
        prompt = self.generate_prompt(query, context)
        
        # 4. Query model
        return self.query_ollama(model_name, prompt)