import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimpleRetriever:
    def __init__(self, docs_path="docs"):
        self.docs_path = docs_path
        self.chunks = []
        self.chunk_ids = []
        self.vectorizer = None
        self.tfidf_matrix = None
        
    def load_documents(self):
        """Load and chunk all documents"""
        self.chunks = []
        self.chunk_ids = []
        
        for filename in os.listdir(self.docs_path):
            if filename.endswith('.md'):
                filepath = os.path.join(self.docs_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple chunking by sections and paragraphs
                sections = re.split(r'\n#+ ', content)
                for i, section in enumerate(sections):
                    if section.strip():
                        # Further split by paragraphs
                        paragraphs = re.split(r'\n\n+', section)
                        for j, para in enumerate(paragraphs):
                            if para.strip() and len(para.strip()) > 10:
                                chunk_id = f"{filename.replace('.md', '')}::chunk{i}_{j}"
                                self.chunks.append(para.strip())
                                self.chunk_ids.append(chunk_id)
        
        # Build TF-IDF index
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
    
    def search(self, query, top_k=3):
        """Search for relevant chunks"""
        if not self.chunks:
            self.load_documents()
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top-k results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                results.append({
                    'content': self.chunks[idx],
                    'chunk_id': self.chunk_ids[idx],
                    'score': float(similarities[idx])
                })
        
        return results

# Global retriever instance
retriever = SimpleRetriever()