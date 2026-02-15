import pandas as pd
import re
import numpy as np
from rapidfuzz import process, fuzz
from typing import List, Dict, Tuple

class MedicalRAG:
    def __init__(self, data_loader=None):
        """
        Système RAG optimisé pour les données médicales
        """
        self.data_loader = data_loader
        self.qa_pairs = []
        self.documents = []
        self.term_to_doc = {}
        
    def build_from_dataframe(self, df: pd.DataFrame):
        """
        Construit l'index RAG à partir d'un DataFrame
        """
        print("🔧 Construction de l'index RAG...")
        
        # 1. Créer des documents à partir des données
        self.documents = []
        self.qa_pairs = []
        
        for idx, row in df.iterrows():
            # Document pour les termes médicaux
            term = str(row.get('term', '')).strip()
            explanation = str(row.get('explanation', '')).strip()
            abbreviation = str(row.get('abbreviation', '')).strip()
            
            if term and explanation:
                doc_text = f"{term}. {explanation}"
                if abbreviation and abbreviation != 'nan':
                    doc_text += f" Abréviation: {abbreviation}"
                
                self.documents.append({
                    'text': doc_text,
                    'term': term,
                    'type': row.get('type', ''),
                    'category': row.get('category', ''),
                    'source': 'dataset'
                })
                
                # Index pour recherche rapide
                self.term_to_doc[term.lower()] = len(self.documents) - 1
                if abbreviation and abbreviation != 'nan':
                    self.term_to_doc[abbreviation.lower()] = len(self.documents) - 1
        
        print(f"   ✓ {len(self.documents)} documents indexés")
        return self
    
    def build_from_data_loader(self, data_loader):
        """
        Construit l'index à partir du data_loader
        """
        self.data_loader = data_loader
        
        # 1. Documents à partir des termes
        for term, data in data_loader.terms_dict.items():
            explanation = data.get('explanation', '')
            if explanation:
                self.documents.append({
                    'text': f"{term}. {explanation}",
                    'term': term,
                    'type': data.get('type', ''),
                    'category': data.get('category', ''),
                    'source': data.get('source', ''),
                    'data': data
                })
        
        # 2. Documents à partir des QA pairs
        for qa in data_loader.qa_pairs:
            self.qa_pairs.append(qa)
            self.documents.append({
                'text': f"Q: {qa['question']} A: {qa['answer']}",
                'term': qa.get('question', '')[:50],
                'type': 'qa_pair',
                'source': qa.get('source', ''),
                'is_qa': True
            })
        
        print(f"🔧 Index RAG construit: {len(self.documents)} documents")
        return self
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Recherche intelligente dans les documents
        """
        query_lower = query.lower().strip()
        results = []
        
        # 1. Recherche exacte des termes
        for doc in self.documents:
            term = doc.get('term', '').lower()
            text = doc.get('text', '').lower()
            
            # Score basé sur plusieurs critères
            score = 0
            
            # a) Terme exact dans la query
            if term and term in query_lower:
                score += 100
            
            # b) Mots-clés dans la query
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 3 and word in text:
                    score += 10
            
            # c) Similarité partielle
            if term and fuzz.partial_ratio(query_lower, term) > 80:
                score += 50
            
            # d) Pour les QA, vérifier si c'est une question
            if doc.get('is_qa', False) and any(q_word in query_lower for q_word in ['quoi', 'comment', 'pourquoi', 'quand']):
                score += 30
            
            if score > 0:
                results.append({
                    'document': doc,
                    'score': score,
                    'text': doc['text'],
                    'source': doc.get('source', ''),
                    'type': doc.get('type', '')
                })
        
        # 2. Trier par score et prendre les meilleurs
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # 3. Si pas de résultats, chercher avec fuzzy matching
        if not results and len(query_lower) > 3:
            all_terms = [doc.get('term', '') for doc in self.documents]
            matches = process.extract(query_lower, all_terms, limit=top_k)
            
            for match, score, idx in matches:
                if score > 60:
                    doc = self.documents[idx]
                    results.append({
                        'document': doc,
                        'score': score,
                        'text': doc['text'],
                        'source': doc.get('source', ''),
                        'type': doc.get('type', '')
                    })
        
        return results[:top_k]
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Formate le contexte pour le prompt IA
        """
        context_parts = []
        
        for i, doc_info in enumerate(retrieved_docs, 1):
            doc = doc_info['document']
            text = doc.get('text', '')
            source = doc.get('source', '')
            
            # Nettoyer le texte
            text = text.replace('Q:', 'Question:').replace('A:', 'Réponse:')
            text = re.sub(r'\s+', ' ', text).strip()
            
            context_parts.append(f"[Document {i} - Source: {source}]\n{text}")
        
        return "\n\n".join(context_parts)

# Instance globale
rag_system = MedicalRAG()

def build_rag_from_loader(data_loader):
    """Fonction helper pour construire le RAG"""
    return rag_system.build_from_data_loader(data_loader)

def retrieve_information(query, top_k=3):
    """Fonction wrapper pour la recherche"""
    return rag_system.retrieve(query, top_k)

def get_rag_context(query, top_k=3):
    """Obtient le contexte formaté pour l'IA"""
    docs = rag_system.retrieve(query, top_k)
    return rag_system.format_context(docs)