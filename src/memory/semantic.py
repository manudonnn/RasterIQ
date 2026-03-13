import json
import os
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticMemory:
    """Loads domain_knowledge.json into a FAISS index to answer queries about domain terminology."""
    
    def __init__(self, knowledge_path="src/knowledge/domain_knowledge.json"):
        self.knowledge_path = knowledge_path
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.entries = []
        self.index = None
        
        self.load_knowledge()

    def load_knowledge(self):
        """Loads definitions from JSON and creates a FAISS vector index."""
        if not os.path.exists(self.knowledge_path):
            print(f"Warning: {self.knowledge_path} not found.")
            return

        with open(self.knowledge_path, "r", encoding="utf-8") as f:
            self.entries = json.load(f)
            
        if not self.entries:
            return

        # Prepare text for embedding: "Concept (Type): Description"
        texts = [
            f"{entry['concept']} ({entry['type']}): {entry['description']}" 
            for entry in self.entries
        ]
        
        # Batch encode
        embeddings = self.encoder.encode(texts)
        
        # Initialize FAISS
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

    def retrieve_concept(self, query: str, top_k: int = 3) -> str:
        """Searches the domain knowledge for the provided query context."""
        if self.index is None or not self.entries:
            return "No semantic domain knowledge loaded."

        query_vector = self.encoder.encode([query]).astype('float32')
        distances, indices = self.index.search(query_vector, min(top_k, len(self.entries)))
        
        results = []
        for i in indices[0]:
            if i != -1:  # valid index
                entry = self.entries[i]
                results.append(f"- {entry['concept']} ({entry['type']}): {entry['description']}")
                
        if results:
            return "Domain Knowledge Retrieved:\n" + "\n".join(results)
        
        return "No relevant domain knowledge found."
