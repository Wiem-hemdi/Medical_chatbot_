# data_loader.py
import pandas as pd
import os
from pathlib import Path

class MedicalDataLoader:
    def __init__(self):
        self.base_path = r"C:\Users\Admin\Desktop\medical_chatbot1\datasets"
        self.dfs = {}
        self.all_data = {}
        
    def load_all_datasets(self):
        """Charge tous les datasets séparément"""
        print("="*60)
        print("📚 CHARGEMENT DE TOUS LES DATASETS")
        print("="*60)
        
        # 1. Dataset principal original
        original_path = os.path.join(self.base_path, 'dataset_medical_fr.csv')
        if os.path.exists(original_path):
            df_original = pd.read_csv(original_path, encoding='utf-8-sig')
            print(f"📊 Dataset original: {len(df_original)} entrées")
            
            # Standardiser les colonnes
            df_original = df_original.rename(columns={
                'terme': 'term',
                'abreviation': 'abbreviation',
                'categorie': 'category',
                'normal_min': 'normal_min',
                'normal_max': 'normal_max',
                'critique_bas': 'critique_bas',
                'critique_haut': 'critique_haut',
                'unite': 'unit',
                'explication': 'explanation',
                'score_frequence': 'frequency_score',
                'code_cim10': 'code_cim10',
                'code_ccam': 'code_ccam',
                'id_snomed': 'id_snomed'
            })
            
            # Nettoyer
            df_original['term'] = df_original['term'].astype(str).str.lower().str.strip()
            df_original['abbreviation'] = df_original['abbreviation'].fillna("").astype(str).str.lower().str.strip()
            df_original['explanation'] = df_original['explanation'].fillna("").astype(str)
            
            self.dfs['original'] = df_original
        else:
            print("❌ Dataset original non trouvé")
            self.dfs['original'] = pd.DataFrame()
        
        # 2. Dataset augmenté 1
        aug1_path = os.path.join(self.base_path, 'augmented', 'medical_qa_20260108_211522.csv')
        if os.path.exists(aug1_path):
            df_aug1 = pd.read_csv(aug1_path, encoding='utf-8-sig')
            print(f"📊 Dataset augmenté 1: {len(df_aug1)} entrées")
            self.dfs['augmented_1'] = df_aug1
        else:
            print("⚠️ Dataset augmenté 1 non trouvé")
            self.dfs['augmented_1'] = pd.DataFrame()
        
        # 3. Dataset augmenté 2
        aug2_path = os.path.join(self.base_path, 'augmented', 'medical_qa_augmented.csv')
        if os.path.exists(aug2_path):
            df_aug2 = pd.read_csv(aug2_path, encoding='utf-8-sig')
            print(f"📊 Dataset augmenté 2: {len(df_aug2)} entrées")
            self.dfs['augmented_2'] = df_aug2
        else:
            print("⚠️ Dataset augmenté 2 non trouvé")
            self.dfs['augmented_2'] = pd.DataFrame()
        
        # 4. Créer des structures de recherche unifiées
        self._create_search_structures()
        
        print(f"✅ Total: {sum(len(df) for df in self.dfs.values())} entrées chargées")
        return self.dfs
    
    def _create_search_structures(self):
        """Crée des structures pour la recherche rapide"""
        
        # Structure pour les termes médicaux
        self.terms_dict = {}
        self.abbreviations_dict = {}
        self.qa_pairs = []  # Paires question-réponse
        
        # 1. Dataset original (analyse labo, maladies, médicaments, etc.)
        if not self.dfs['original'].empty:
            for _, row in self.dfs['original'].iterrows():
                term = str(row.get('term', '')).lower().strip()
                if term:
                    self.terms_dict[term] = {
                        'source': 'original',
                        'type': row.get('type'),
                        'category': row.get('category'),
                        'explanation': row.get('explanation'),
                        'normal_min': row.get('normal_min'),
                        'normal_max': row.get('normal_max'),
                        'unit': row.get('unit'),
                        'abbreviation': row.get('abbreviation'),
                        'row_data': row.to_dict()
                    }
                    
                    # Ajouter l'abréviation si elle existe
                    abbrev = str(row.get('abbreviation', '')).lower().strip()
                    if abbrev and abbrev not in ['nan', '']:
                        self.abbreviations_dict[abbrev] = term
        
        # 2. Dataset augmenté 1 (QA)
        if not self.dfs['augmented_1'].empty:
            for _, row in self.dfs['augmented_1'].iterrows():
                # Chercher les paires Q/R
                if 'question' in row and 'answer' in row:
                    question = str(row['question']).strip()
                    answer = str(row['answer']).strip()
                    if question and answer:
                        self.qa_pairs.append({
                            'question': question,
                            'answer': answer,
                            'source': 'augmented_1'
                        })
                
                # Chercher les termes
                for col in ['term', 'terme', 'lab_name', 'nom']:
                    if col in row and pd.notna(row[col]):
                        term = str(row[col]).lower().strip()
                        if term and term not in self.terms_dict:
                            self.terms_dict[term] = {
                                'source': 'augmented_1',
                                'type': row.get('type', 'qa_pair'),
                                'explanation': row.get('explanation') or row.get('answer', ''),
                                'row_data': row.to_dict()
                            }
        
        # 3. Dataset augmenté 2 (QA)
        if not self.dfs['augmented_2'].empty:
            for _, row in self.dfs['augmented_2'].iterrows():
                # Chercher les paires Q/R
                if 'question' in row and 'answer' in row:
                    question = str(row['question']).strip()
                    answer = str(row['answer']).strip()
                    if question and answer:
                        self.qa_pairs.append({
                            'question': question,
                            'answer': answer,
                            'source': 'augmented_2'
                        })
                
                # Chercher les termes
                for col in ['term', 'terme', 'lab_name', 'nom']:
                    if col in row and pd.notna(row[col]):
                        term = str(row[col]).lower().strip()
                        if term and term not in self.terms_dict:
                            self.terms_dict[term] = {
                                'source': 'augmented_2',
                                'type': row.get('type', 'qa_pair'),
                                'explanation': row.get('explanation') or row.get('answer', ''),
                                'row_data': row.to_dict()
                            }
        
        print(f"📖 Index créé: {len(self.terms_dict)} termes, {len(self.qa_pairs)} paires Q/R")
    
    def search_term(self, term):
        """Recherche un terme dans tous les datasets"""
        term_lower = term.lower().strip()
        
        # 1. Chercher dans les termes exacts
        if term_lower in self.terms_dict:
            return self.terms_dict[term_lower]
        
        # 2. Chercher dans les abréviations
        if term_lower in self.abbreviations_dict:
            actual_term = self.abbreviations_dict[term_lower]
            return self.terms_dict.get(actual_term)
        
        # 3. Chercher partiellement
        for stored_term, data in self.terms_dict.items():
            if term_lower in stored_term or stored_term in term_lower:
                return data
        
        return None
    
    def search_qa(self, question):
        """Recherche une réponse dans les paires Q/R"""
        question_lower = question.lower().strip()
        
        # Recherche exacte
        for qa in self.qa_pairs:
            if qa['question'].lower() == question_lower:
                return qa['answer']
        
        # Recherche partielle
        for qa in self.qa_pairs:
            q_lower = qa['question'].lower()
            # Si la question contient des mots-clés de la recherche
            if any(word in q_lower for word in question_lower.split() if len(word) > 3):
                return qa['answer']
        
        return None
    
    def get_all_terms(self):
        """Retourne tous les termes pour la recherche floue"""
        return list(self.terms_dict.keys())
    
    def get_all_abbreviations(self):
        """Retourne toutes les abréviations"""
        return list(self.abbreviations_dict.keys())

# Singleton pour charger une fois
data_loader = MedicalDataLoader()