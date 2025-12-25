
# enhanced_rag_system.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class EnhancedMedicalRAG:
    def __init__(self, dataset_path):
        # Charger les données augmentées
        self.df = pd.read_csv(dataset_path)
        self.df['term'] = self.df['term'].str.lower()
        self.df['abbreviation'] = self.df['abbreviation'].fillna("").str.lower()
        
        # Préparer le texte pour la recherche
        self.df['search_text'] = self.df.apply(
            lambda row: f"{row['term']} {row['abbreviation']} {row['category']} {row['explanation']}",
            axis=1
        )
        
        # Initialiser TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words=['le', 'la', 'les', 'de', 'des', 'du', 'et', 'est'],
            ngram_range=(1, 3)  # Unigrammes, bigrammes, trigrammes
        )
        
        self.X = self.vectorizer.fit_transform(self.df['search_text'])
        
        # Charger les paires Q-R
        try:
            self.qa_df = pd.read_csv(dataset_path.replace('dataset', 'qa_pairs'))
            self.has_qa = True
        except:
            self.has_qa = False
    
    def retrieve_context(self, query, top_k=3):
        """Recherche améliorée avec matching sémantique"""
        query_vec = self.vectorizer.transform([query.lower()])
        similarities = cosine_similarity(query_vec, self.X)[0]
        
        # Trouver les meilleures correspondances
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.15:  # Seuil de pertinence
                row = self.df.iloc[idx]
                result = {
                    'term': row['term'],
                    'abbreviation': row['abbreviation'],
                    'type': row['type'],
                    'category': row['category'],
                    'explanation': row['explanation'],
                    'similarity': similarities[idx]
                }
                
                # Ajouter des valeurs normales si disponibles
                if row['type'] == 'lab test' and pd.notna(row['normal_min']):
                    result['normal_range'] = f"{row['normal_min']}-{row['normal_max']} {row['unit']}"
                
                results.append(result)
        
        return results
    
    def find_qa_match(self, query):
        """Cherche une correspondance dans les paires Q-R"""
        if not self.has_qa:
            return None
        
        # Recherche par similarité de texte
        for _, qa_row in self.qa_df.iterrows():
            if qa_row['question'].lower() in query.lower() or query.lower() in qa_row['question'].lower():
                return qa_row['answer']
        
        return None
    
    def generate_context_prompt(self, query):
        """Génère un prompt enrichi pour Ollama"""
        # 1. Chercher dans Q-R d'abord
        qa_answer = self.find_qa_match(query)
        if qa_answer:
            return f"""QUESTION: {query}

RÉPONSE PRÉ-ENTRAÎNÉE: {qa_answer}

(Pourriez-vous reformuler cette réponse de manière plus naturelle pour un patient?)"""
        
        # 2. Recherche RAG standard
        contexts = self.retrieve_context(query)
        
        if not contexts:
            return f"QUESTION: {query}\n\n(Je n'ai pas d'information spécifique sur ce sujet)"
        
        # Construire le contexte
        context_text = "\n".join([
            f"- {ctx['term'].title()} ({ctx.get('abbreviation', '')}): {ctx['explanation']} "
            f"{f'Plage normale: {ctx.get("normal_range", "")}' if 'normal_range' in ctx else ''}"
            for ctx in contexts
        ])
        
        return f"""CONTEXTE MÉDICAL:
{context_text}

QUESTION DU PATIENT: {query}

INSTRUCTIONS:
1. Utilisez le contexte médical ci-dessus
2. Répondez en français simple
3. Soyez précis mais accessible
4. Mentionnez les valeurs de référence si disponibles
5. Recommandez de consulter un médecin si nécessaire

RÉPONSE:"""

# Utilisation dans votre chatbot
def integrate_with_chatbot():
    # Initialiser le RAG amélioré
    medical_rag = EnhancedMedicalRAG("datasets/medical_dataset_augmented.csv")
    
    def enhanced_ask_phi(question):
        # Générer le prompt enrichi
        prompt = medical_rag.generate_context_prompt(question)
        
        # Appeler Ollama avec le prompt
        # ... votre code Ollama existant ...
        
        return response
    
    return enhanced_ask_phi
