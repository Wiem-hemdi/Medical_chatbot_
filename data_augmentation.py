# data_augmentation.py
import pandas as pd
import numpy as np
import re
import random
from datetime import datetime
import json

print("=" * 60)
print("🔬 AUGMENTATION DE DONNÉES MÉDICALES")
print("=" * 60)

# -----------------------------
# 1. Charger votre dataset
# -----------------------------
df = pd.read_csv("datasets/comprehensive_medical_dataset.csv")
print(f"📊 Dataset original: {len(df)} entrées")

# -----------------------------
# 2. Fonctions d'augmentation
# -----------------------------
def augment_explanations(text, term_type, category):
    """Augmente les explications avec des variantes"""
    augmentations = []
    
    if term_type == "lab test":
        patterns = [
            f"Le test {text} mesure",
            f"Résultat de {text} indique",
            f"Valeur de référence pour {text}",
            f"Interprétation clinique du {text}",
            f"{text} - examen de laboratoire",
            f"Analyse sanguine: {text}"
        ]
    elif term_type == "disease":
        patterns = [
            f"La maladie {text} est",
            f"Pathologie: {text}",
            f"Trouble médical: {text}",
            f"Condition de santé: {text}",
            f"Diagnostic de {text}",
            f"Symptômes du {text}"
        ]
    elif term_type == "medication":
        patterns = [
            f"Médicament {text} prescrit pour",
            f"Traitement par {text}",
            f"Thérapie médicamenteuse: {text}",
            f"Posologie du {text}",
            f"Effets du {text}",
            f"Prescription de {text}"
        ]
    else:
        patterns = [f"{text} - "]
    
    return [random.choice(patterns) + " " + text for _ in range(3)]

def generate_synonyms(term):
    """Génère des synonymes et variantes"""
    synonyms_dict = {
        # Tests labo
        "Hemoglobin": ["Hémoglobine", "Taux d'Hb", "Hb sanguine", "Hémoglobine totale"],
        "White Blood Cell Count": ["Leucocytes", "Numération globulaire blanche", "GB", "WBC"],
        "Platelet Count": ["Plaquettes", "Thrombocytes", "Numération plaquettaire"],
        "Cholesterol Total": ["Cholestérol total", "CT", "Cholestérol sanguin"],
        "Glucose": ["Glycémie", "Sucre sanguin", "Glucose sanguin"],
        
        # Maladies
        "Hypertension": ["HTA", "Pression artérielle élevée", "Hypertension artérielle"],
        "Diabetes": ["Diabète sucré", "Maladie diabétique", "Hyperglycémie chronique"],
        "Asthma": ["Asthme bronchique", "Crise d'asthme", "Affection respiratoire"],
        
        # Symptômes
        "Headache": ["Céphalée", "Mal de tête", "Migraine", "Douleur crânienne"],
        "Fever": ["Fièvre", "Pyrexie", "Hyperthermie", "Température élevée"],
        "Fatigue": ["Fatigue", "Asthénie", "Épuisement", "Lassitude"],
    }
    
    return synonyms_dict.get(term, [term + " (variante)"])

def create_question_variants(term, explanation, term_type):
    """Crée des questions variées pour le terme"""
    questions = []
    
    base_questions = [
        f"Qu'est-ce que {term}?",
        f"Quelle est la signification de {term}?",
        f"Définition de {term}",
        f"Expliquez {term}",
        f"Que signifie {term} en médecine?",
        f"Rôle de {term}",
        f"Importance de {term}",
    ]
    
    if term_type == "lab test":
        questions.extend([
            f"Valeurs normales pour {term}",
            f"Interprétation du test {term}",
            f"{term} trop élevé, que faire?",
            f"{term} trop bas, causes?",
            f"Quand prescrire {term}?",
        ])
    elif term_type == "disease":
        questions.extend([
            f"Symptômes du {term}",
            f"Traitement de {term}",
            f"Causes de {term}",
            f"Diagnostic du {term}",
            f"Prévention du {term}",
        ])
    
    return questions

def augment_with_contextual_info(row):
    """Ajoute des informations contextuelles"""
    augmented = []
    
    if row['type'] == 'lab test' and pd.notna(row['normal_min']) and pd.notna(row['normal_max']):
        # Variantes de plages normales
        contexts = [
            f"Plage de référence: {row['normal_min']}-{row['normal_max']} {row['unit']}",
            f"Valeurs normales: entre {row['normal_min']} et {row['normal_max']} {row['unit']}",
            f"Intervalle physiologique: {row['normal_min']} à {row['normal_max']} {row['unit']}",
            f"Seuils: normal = {row['normal_min']}-{row['normal_max']}, critique bas = {row.get('critical_low', 'N/A')}, critique haut = {row.get('critical_high', 'N/A')}",
        ]
        augmented.extend(contexts)
    
    if row['category']:
        # Contexte par catégorie
        category_context = {
            'Hematology': ["Hématologie - étude du sang", "Système sanguin"],
            'Biochemistry': ["Biochimie sanguine", "Métabolisme"],
            'Endocrinology': ["Système endocrinien", "Hormones"],
            'Cardiovascular': ["Système cardiovasculaire", "Cœur et vaisseaux"],
            'Respiratory': ["Appareil respiratoire", "Poumons"],
        }
        if row['category'] in category_context:
            augmented.extend(category_context[row['category']])
    
    return augmented

# -----------------------------
# 3. Application de l'augmentation
# -----------------------------
def augment_dataset(df):
    """Applique toutes les techniques d'augmentation"""
    augmented_rows = []
    
    for idx, row in df.iterrows():
        # Version originale
        augmented_rows.append(row.to_dict())
        
        # 1. Synonymes et variantes terminologiques
        synonyms = generate_synonyms(row['term'])
        for syn in synonyms[:2]:  # Prend 2 synonymes max
            if syn != row['term']:
                new_row = row.copy()
                new_row['term'] = syn
                new_row['abbreviation'] = f"{row['abbreviation']}_syn" if pd.notna(row['abbreviation']) else ""
                augmented_rows.append(new_row.to_dict())
        
        # 2. Explications augmentées
        explanations = augment_explanations(row['explanation'], row['type'], row['category'])
        for exp in explanations[:2]:  # 2 variantes d'explication
            new_row = row.copy()
            new_row['explanation'] = exp
            new_row['frequency_score'] = row['frequency_score'] * 0.8  # Légère réduction du score
            augmented_rows.append(new_row.to_dict())
        
        # 3. Contextes cliniques
        contexts = augment_with_contextual_info(row)
        for ctx in contexts[:2]:  # 2 contextes max
            new_row = row.copy()
            new_row['explanation'] = f"{row['explanation']} {ctx}"
            augmented_rows.append(new_row.to_dict())
    
    return pd.DataFrame(augmented_rows)

# -----------------------------
# 4. Génération de données synthétiques
# -----------------------------
def generate_synthetic_medical_data(base_df):
    """Génère des données médicales synthétiques basées sur les patterns"""
    synthetic = []
    
    # Patterns pour chaque type
    lab_test_patterns = [
        "{term} ({abbreviation}) - Test de laboratoire en {category}",
        "Mesure du {term} - Indicateur clinique important",
        "{term}: examen sanguin pour évaluer {category}",
    ]
    
    disease_patterns = [
        "{term} ({abbreviation}) - Maladie du système {category}",
        "Pathologie: {term} - Affecte le système {category}",
        "{term}: trouble médical nécessitant un suivi",
    ]
    
    for _, row in base_df.iterrows():
        if row['type'] == 'lab test' and random.random() > 0.7:
            # Génère des variations de plages normales
            for _ in range(2):
                new_row = row.copy()
                # Légère variation des valeurs normales (±10%)
                if pd.notna(row['normal_min']) and pd.notna(row['normal_max']):
                    variation = random.uniform(0.9, 1.1)
                    new_row['normal_min'] = float(row['normal_min']) * variation
                    new_row['normal_max'] = float(row['normal_max']) * variation
                
                # Variation d'unités équivalentes
                unit_variants = {
                    'g/dL': ['g/L', 'mmol/L'],
                    'mmol/L': ['mg/dL', 'g/L'],
                    'mg/L': ['g/L', 'μg/mL'],
                }
                if row['unit'] in unit_variants:
                    new_row['unit'] = random.choice(unit_variants[row['unit']])
                
                synthetic.append(new_row.to_dict())
    
    return pd.DataFrame(synthetic)

# -----------------------------
# 5. Création de paires Question-Réponse
# -----------------------------
def create_qa_pairs(df):
    """Crée des paires question-réponse pour l'entraînement"""
    qa_data = []
    
    for _, row in df.iterrows():
        term = row['term']
        explanation = row['explanation']
        term_type = row['type']
        
        # Questions de base
        questions = create_question_variants(term, explanation, term_type)
        
        # Réponses enrichies
        base_answer = f"{term}: {explanation}"
        
        if row['type'] == 'lab test' and pd.notna(row['normal_min']):
            answer_variants = [
                f"{base_answer} Valeurs normales: {row['normal_min']}-{row['normal_max']} {row['unit']}.",
                f"Le test {term} mesure {explanation.lower()}. La plage normale est de {row['normal_min']} à {row['normal_max']} {row['unit']}.",
                f"{term} ({row.get('abbreviation', '')}): {explanation} Référence: {row['normal_min']}-{row['normal_max']} {row['unit']}.",
            ]
        else:
            answer_variants = [
                base_answer,
                f"En médecine, {term} se réfère à: {explanation}",
                f"Définition: {term} - {explanation}",
            ]
        
        # Créer des paires
        for question in questions[:3]:  # 3 questions par terme
            for answer in answer_variants[:2]:  # 2 réponses par question
                qa_data.append({
                    'question': question,
                    'answer': answer,
                    'term': term,
                    'type': term_type,
                    'category': row['category'],
                    'source': 'augmented'
                })
    
    return pd.DataFrame(qa_data)

# -----------------------------
# 6. Exécution principale
# -----------------------------
def main():
    print("🚀 Début de l'augmentation des données...")
    
    # Augmentation du dataset principal
    augmented_df = augment_dataset(df)
    print(f"✅ Dataset augmenté: {len(augmented_df)} entrées")
    
    # Génération de données synthétiques
    synthetic_df = generate_synthetic_medical_data(df)
    print(f"✅ Données synthétiques: {len(synthetic_df)} entrées")
    
    # Création de paires Q-R
    qa_df = create_qa_pairs(df)
    print(f"✅ Paires Question-Réponse: {len(qa_df)} paires")
    
    # Fusion de tous les datasets
    final_df = pd.concat([augmented_df, synthetic_df], ignore_index=True)
    final_df = final_df.drop_duplicates(subset=['term', 'explanation'], keep='first')
    
    # Sauvegarde
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Dataset médical augmenté
    final_df.to_csv(f"datasets/medical_dataset_augmented_{timestamp}.csv", index=False)
    
    # 2. Paires Q-R pour l'entraînement RAG
    qa_df.to_csv(f"datasets/medical_qa_pairs_{timestamp}.csv", index=False)
    
    # 3. Fichier JSON pour Ollama fine-tuning
    qa_json = qa_df[['question', 'answer']].to_dict('records')
    with open(f"datasets/medical_qa_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump(qa_json, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("🎉 AUGMENTATION TERMINÉE !")
    print("=" * 60)
    print(f"📁 Fichiers créés:")
    print(f"   1. medical_dataset_augmented_{timestamp}.csv")
    print(f"   2. medical_qa_pairs_{timestamp}.csv")
    print(f"   3. medical_qa_{timestamp}.json")
    print(f"\n📈 Statistiques:")
    print(f"   - Dataset original: {len(df)} entrées")
    print(f"   - Dataset final: {len(final_df)} entrées")
    print(f"   - Paires Q-R: {len(qa_df)} paires")
    print(f"   - Augmentation: {len(final_df)/len(df):.1f}x")
    
    # Aperçu des données générées
    print(f"\n🔍 Aperçu des données augmentées:")
    sample = final_df[['term', 'type', 'category', 'explanation']].head(5)
    for _, row in sample.iterrows():
        print(f"   • {row['term']} ({row['type']}): {row['explanation'][:80]}...")

# -----------------------------
# 7. Script d'intégration avec votre chatbot
# -----------------------------
def create_enhanced_rag_system():
    """Crée un système RAG amélioré avec les données augmentées"""
    
    code = '''
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
            return f"QUESTION: {query}\\n\\n(Je n'ai pas d'information spécifique sur ce sujet)"
        
        # Construire le contexte
        context_text = "\\n".join([
            f"- {ctx['term'].title()} ({ctx.get('abbreviation', '')}): {ctx['explanation']} "
            f"{f'Plage normale: {ctx.get(\"normal_range\", \"\")}' if 'normal_range' in ctx else ''}"
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
'''
    
    # Sauvegarder le code
    with open("enhanced_rag_system.py", "w", encoding="utf-8") as f:
        f.write(code)
    
    print(f"\n💡 Système RAG amélioré créé: enhanced_rag_system.py")

if __name__ == "__main__":
    # Exécuter l'augmentation
    main()
    
    # Créer le système RAG amélioré
    create_enhanced_rag_system()