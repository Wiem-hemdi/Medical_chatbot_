"""
Script de Data Augmentation Q&A pour Chatbot Médical - Version Complète
Génère des paires Question-Réponse diversifiées avec validation médicale
Version 3.0 - Janvier 2026

AVERTISSEMENT: Validation médicale professionnelle requise avant usage clinique
"""

import pandas as pd
import numpy as np
import random
import os
import json
from typing import List, Dict, Tuple
from datetime import datetime
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Configuration des chemins
CHEMIN_ENTREE = r"C:\Users\Admin\Desktop\medical_chatbot1\datasets\dataset_medical_fr.csv"
DOSSIER_SORTIE = r"C:\Users\Admin\Desktop\medical_chatbot1\datasets\augmented"

# Paramètres
VARIATIONS_PAR_TERME = 10
SEED = 42

# Seeds pour reproductibilité
np.random.seed(SEED)
random.seed(SEED)

# ============================================================================
# TEMPLATES DE QUESTIONS - VERSION COMPLÈTE
# ============================================================================

TEMPLATES_QUESTIONS = {
    'definition': [
        "Qu'est-ce que {terme} ?",
        "Peux-tu m'expliquer ce qu'est {terme} ?",
        "Définis {terme}",
        "C'est quoi {terme} ?",
        "Explique-moi {terme}",
        "Je voudrais comprendre ce qu'est {terme}",
        "Donne-moi la définition de {terme}",
        "Que signifie {terme} ?",
        "Peux-tu me dire ce qu'est {terme} ?",
        "J'aimerais savoir ce qu'est {terme}",
        "Aide-moi à comprendre {terme}",
        "Qu'entend-on par {terme} ?",
        "Pourrais-tu expliquer {terme} ?",
        "{terme}, qu'est-ce que c'est exactement ?",
        "Je ne connais pas {terme}, peux-tu m'aider ?",
        "En médecine, qu'est-ce que {terme} ?",
        "Dans le contexte médical, que signifie {terme} ?",
    ],
    
    'abbreviation': [
        "Que signifie l'abréviation {abrev} ?",
        "C'est quoi {abrev} ?",
        "{abrev} signifie quoi ?",
        "Quelle est la signification de {abrev} ?",
        "Explique {abrev}",
        "{abrev} veut dire quoi ?",
        "Que veut dire {abrev} en médecine ?",
        "Peux-tu m'expliquer {abrev} ?",
        "Qu'est-ce que {abrev} signifie ?",
        "J'ai vu {abrev} sur mes analyses, c'est quoi ?",
        "Mon médecin a parlé de {abrev}, qu'est-ce que c'est ?",
    ],
    
    'valeurs_normales': [
        "Quelles sont les valeurs normales de {terme} ?",
        "Quelle est la plage normale pour {terme} ?",
        "Valeurs de référence de {terme} ?",
        "Quels sont les taux normaux de {terme} ?",
        "C'est quoi les normes pour {terme} ?",
        "Quelle est la norme de {terme} ?",
        "Donne-moi les valeurs normales de {terme}",
        "Plage de référence pour {terme} ?",
        "Valeurs standards de {terme} ?",
        "Quel est le taux normal de {terme} ?",
        "Entre quelles valeurs {terme} doit se situer ?",
        "Mon {terme} est à combien normalement ?",
    ],
    
    'valeurs_critiques': [
        "Quelles sont les valeurs critiques de {terme} ?",
        "À partir de quand {terme} devient dangereux ?",
        "Valeurs alarmantes de {terme} ?",
        "Seuils critiques pour {terme} ?",
        "Quand s'inquiéter pour {terme} ?",
        "À quel niveau {terme} devient préoccupant ?",
        "Limites dangereuses de {terme} ?",
    ],
    
    'interpretation': [
        "J'ai {terme} à {valeur} {unite}, c'est normal ?",
        "Mon analyse montre {terme} = {valeur} {unite}, qu'en penses-tu ?",
        "Est-ce que {valeur} {unite} pour {terme} est bon ?",
        "{terme} à {valeur} {unite}, est-ce inquiétant ?",
    ],
    
    'symptomes': [
        "Quels sont les symptômes de {terme} ?",
        "Comment se manifeste {terme} ?",
        "Signes de {terme} ?",
        "Comment reconnaître {terme} ?",
        "Quels sont les signes de {terme} ?",
        "Comment savoir si j'ai {terme} ?",
        "Manifestations de {terme} ?",
        "Symptômes typiques de {terme} ?",
    ],
    
    'traitement': [
        "Comment traiter {terme} ?",
        "Quel est le traitement pour {terme} ?",
        "Comment soigner {terme} ?",
        "Thérapie pour {terme} ?",
        "Quel traitement pour {terme} ?",
        "Prise en charge de {terme} ?",
        "Options thérapeutiques pour {terme} ?",
    ],
    
    'causes': [
        "Quelles sont les causes de {terme} ?",
        "Pourquoi a-t-on {terme} ?",
        "Qu'est-ce qui provoque {terme} ?",
        "Origines de {terme} ?",
        "Facteurs de risque de {terme} ?",
    ],
    
    'prevention': [
        "Comment prévenir {terme} ?",
        "Peut-on éviter {terme} ?",
        "Prévention de {terme} ?",
    ],
    
    'medicament_usage': [
        "À quoi sert {terme} ?",
        "Pourquoi prendre {terme} ?",
        "Indications de {terme} ?",
        "Dans quels cas utilise-t-on {terme} ?",
    ],
    
    'procedure_deroulement': [
        "Comment se déroule {terme} ?",
        "En quoi consiste {terme} ?",
        "Déroulement de {terme} ?",
    ],
}

# ============================================================================
# AVERTISSEMENTS MÉDICAUX
# ============================================================================

AVERTISSEMENTS = {
    'interpretation': "⚠️ Important : L'interprétation de résultats d'analyses doit être faite par votre médecin en tenant compte de votre contexte médical complet.",
    'diagnostic': "⚠️ Important : Seul un professionnel de santé qualifié peut établir un diagnostic. Cette information est fournie à titre éducatif uniquement.",
    'traitement': "⚠️ Important : Ne modifiez jamais votre traitement sans l'avis de votre médecin. Cette information est fournie à titre éducatif uniquement.",
    'urgence': "🚨 En cas de symptômes graves, contactez immédiatement les services d'urgence (15, 112) ou rendez-vous aux urgences.",
}

# ============================================================================
# CLASSE PRINCIPALE
# ============================================================================

class AugmentateurMedical:
    """Augmentation complète de données médicales Q&A"""
    
    def __init__(self, variations_par_terme: int = 10):
        self.variations_par_terme = variations_par_terme
        self.stats = {
            'total_paires': 0,
            'par_type': {},
            'par_type_question': {},
            'avertissements': 0,
        }
    
    def choisir_templates(self, templates: List[str], n: int) -> List[str]:
        """Choisit n templates aléatoirement"""
        return random.sample(templates, min(n, len(templates)))
    
    def generer_valeur_exemple(self, row: pd.Series) -> float:
        """Génère une valeur d'exemple dans la plage normale"""
        try:
            min_val = float(row['normal_min'])
            max_val = float(row['normal_max'])
            return round(random.uniform(min_val, max_val), 2)
        except:
            return None
    
    def formater_texte(self, texte: str, data: Dict) -> str:
        """Formate un texte avec gestion des valeurs manquantes"""
        data_clean = {}
        for k, v in data.items():
            if pd.isna(v) or v == '':
                data_clean[k] = ''
            else:
                data_clean[k] = str(v)
        
        try:
            result = texte.format(**data_clean)
            # Nettoyer les espaces doubles
            result = ' '.join(result.split())
            return result
        except:
            return texte
    
    def generer_reponse_definition(self, row: pd.Series) -> str:
        """Génère une réponse de définition complète"""
        reponse = ""
        
        # Terme et abréviation
        terme = row['terme']
        abrev = f" ({row['abreviation']})" if pd.notna(row['abreviation']) and row['abreviation'] else ""
        
        # Type et catégorie
        type_dict = {
            'analyse_labo': 'une analyse de laboratoire',
            'maladie': 'une pathologie',
            'procedure': 'une procédure médicale',
            'medicament': 'un médicament',
            'symptome': 'un symptôme'
        }
        
        type_desc = type_dict.get(row['type'], 'un terme médical')
        categorie = f" en {row['categorie']}" if pd.notna(row['categorie']) else ""
        
        reponse = f"{terme}{abrev} est {type_desc}{categorie}. "
        
        # Explication
        if pd.notna(row['explication']):
            reponse += str(row['explication'])
        
        # Valeurs normales pour analyses
        if row['type'] == 'analyse_labo' and pd.notna(row['normal_min']):
            unite = row['unite'] if pd.notna(row['unite']) else ''
            reponse += f" Les valeurs normales se situent entre {row['normal_min']} et {row['normal_max']} {unite}."
        
        return reponse
    
    def generer_paires_definition(self, row: pd.Series) -> List[Dict]:
        """Génère des paires Q&A de définition"""
        paires = []
        questions = self.choisir_templates(TEMPLATES_QUESTIONS['definition'], 3)
        reponse = self.generer_reponse_definition(row)
        
        for question in questions:
            paires.append({
                'question': question.format(terme=row['terme']),
                'reponse': reponse,
                'terme_original': row['terme'],
                'type': row['type'],
                'categorie': row['categorie'],
                'type_question': 'definition',
                'necessite_avertissement': False,
            })
        
        return paires
    
    def generer_paires_abbreviation(self, row: pd.Series) -> List[Dict]:
        """Génère des paires Q&A sur les abréviations"""
        paires = []
        
        if not pd.notna(row['abreviation']) or row['abreviation'] == '':
            return paires
        
        questions = self.choisir_templates(TEMPLATES_QUESTIONS['abbreviation'], 2)
        reponse = f"{row['abreviation']} signifie {row['terme']}."
        
        if pd.notna(row['categorie']):
            reponse += f" C'est un terme utilisé en {row['categorie']}."
        
        for question in questions:
            paires.append({
                'question': question.format(abrev=row['abreviation']),
                'reponse': reponse,
                'terme_original': row['terme'],
                'type': row['type'],
                'categorie': row['categorie'],
                'type_question': 'abbreviation',
                'necessite_avertissement': False,
            })
        
        return paires
    
    def generer_paires_valeurs_normales(self, row: pd.Series) -> List[Dict]:
        """Génère des paires Q&A sur les valeurs normales"""
        paires = []
        
        if row['type'] != 'analyse_labo' or not pd.notna(row['normal_min']):
            return paires
        
        questions = self.choisir_templates(TEMPLATES_QUESTIONS['valeurs_normales'], 2)
        unite = row['unite'] if pd.notna(row['unite']) else ''
        reponse = f"Les valeurs normales de {row['terme']} se situent entre {row['normal_min']} et {row['normal_max']} {unite}."
        
        for question in questions:
            paires.append({
                'question': question.format(terme=row['terme']),
                'reponse': reponse,
                'terme_original': row['terme'],
                'type': row['type'],
                'categorie': row['categorie'],
                'type_question': 'valeurs_normales',
                'necessite_avertissement': True,
            })
        
        return paires
    
    def generer_paires_valeurs_critiques(self, row: pd.Series) -> List[Dict]:
        """Génère des paires Q&A sur les valeurs critiques"""
        paires = []
        
        if row['type'] != 'analyse_labo':
            return paires
        
        if not (pd.notna(row.get('critique_bas')) or pd.notna(row.get('critique_haut'))):
            return paires
        
        questions = self.choisir_templates(TEMPLATES_QUESTIONS['valeurs_critiques'], 1)
        unite = row['unite'] if pd.notna(row['unite']) else ''
        
        reponse = f"Pour {row['terme']}, les valeurs critiques sont : "
        
        parties = []
        if pd.notna(row.get('critique_bas')):
            parties.append(f"en dessous de {row['critique_bas']} {unite} (seuil bas critique)")
        if pd.notna(row.get('critique_haut')):
            parties.append(f"au-dessus de {row['critique_haut']} {unite} (seuil haut critique)")
        
        reponse += " et ".join(parties) + ". "
        reponse += "Ces valeurs nécessitent une attention médicale urgente."
        
        for question in questions:
            paires.append({
                'question': question.format(terme=row['terme']),
                'reponse': reponse,
                'terme_original': row['terme'],
                'type': row['type'],
                'categorie': row['categorie'],
                'type_question': 'valeurs_critiques',
                'necessite_avertissement': True,
            })
        
        return paires
    
    def generer_paires_interpretation(self, row: pd.Series) -> List[Dict]:
        """Génère des paires Q&A d'interprétation avec valeurs"""
        paires = []
        
        if row['type'] != 'analyse_labo' or not pd.notna(row['normal_min']):
            return paires
        
        # Générer 1-2 exemples
        for _ in range(random.randint(1, 2)):
            valeur = self.generer_valeur_exemple(row)
            if not valeur:
                continue
            
            question_template = random.choice(TEMPLATES_QUESTIONS['interpretation'])
            unite = row['unite'] if pd.notna(row['unite']) else ''
            
            question = question_template.format(
                terme=row['terme'],
                valeur=valeur,
                unite=unite
            )
            
            reponse = f"Une valeur de {valeur} {unite} pour {row['terme']} se situe dans la plage normale "
            reponse += f"({row['normal_min']}-{row['normal_max']} {unite}). "
            reponse += "Cependant, seul votre médecin peut interpréter ce résultat dans votre contexte clinique complet."
            
            paires.append({
                'question': question,
                'reponse': reponse,
                'terme_original': row['terme'],
                'type': row['type'],
                'categorie': row['categorie'],
                'type_question': 'interpretation',
                'necessite_avertissement': True,
            })
        
        return paires
    
    def generer_paires_symptomes(self, row: pd.Series) -> List[Dict]:
        """Génère des paires Q&A sur les symptômes"""
        paires = []
        
        if row['type'] != 'maladie':
            return paires
        
        questions = self.choisir_templates(TEMPLATES_QUESTIONS['symptomes'], 2)
        reponse = f"Les symptômes de {row['terme']} peuvent varier selon les individus. "
        reponse += "Il est essentiel de consulter un professionnel de santé pour un diagnostic précis basé sur votre situation personnelle."
        
        for question in questions:
            paires.append({
                'question': question.format(terme=row['terme']),
                'reponse': reponse,
                'terme_original': row['terme'],
                'type': row['type'],
                'categorie': row['categorie'],
                'type_question': 'symptomes',
                'necessite_avertissement': True,
            })
        
        return paires
    
    def generer_paires_traitement(self, row: pd.Series) -> List[Dict]:
        """Génère des paires Q&A sur le traitement"""
        paires = []
        
        if row['type'] not in ['maladie', 'symptome']:
            return paires
        
        questions = self.choisir_templates(TEMPLATES_QUESTIONS['traitement'], 1)
        reponse = f"Le traitement de {row['terme']} doit être personnalisé selon chaque patient. "
        reponse += "Votre médecin déterminera la meilleure approche thérapeutique adaptée à votre situation."
        
        for question in questions:
            paires.append({
                'question': question.format(terme=row['terme']),
                'reponse': reponse,
                'terme_original': row['terme'],
                'type': row['type'],
                'categorie': row['categorie'],
                'type_question': 'traitement',
                'necessite_avertissement': True,
            })
        
        return paires
    
    def generer_paires_causes(self, row: pd.Series) -> List[Dict]:
        """Génère des paires Q&A sur les causes"""
        paires = []
        
        if row['type'] != 'maladie':
            return paires
        
        questions = self.choisir_templates(TEMPLATES_QUESTIONS['causes'], 1)
        reponse = f"Les causes de {row['terme']} peuvent être multiples et varient d'une personne à l'autre. "
        reponse += "Une évaluation médicale permet d'identifier les facteurs spécifiques dans votre cas."
        
        for question in questions:
            paires.append({
                'question': question.format(terme=row['terme']),
                'reponse': reponse,
                'terme_original': row['terme'],
                'type': row['type'],
                'categorie': row['categorie'],
                'type_question': 'causes',
                'necessite_avertissement': False,
            })
        
        return paires
    
    def generer_paires_medicament(self, row: pd.Series) -> List[Dict]:
        """Génère des paires Q&A pour les médicaments"""
        paires = []
        
        if row['type'] != 'medicament':
            return paires
        
        questions = self.choisir_templates(TEMPLATES_QUESTIONS['medicament_usage'], 1)
        reponse = f"{row['terme']} est un médicament"
        
        if pd.notna(row['categorie']):
            reponse += f" de la classe {row['categorie']}"
        
        reponse += ". Votre médecin déterminera si ce médicament est approprié pour votre situation."
        
        for question in questions:
            paires.append({
                'question': question.format(terme=row['terme']),
                'reponse': reponse,
                'terme_original': row['terme'],
                'type': row['type'],
                'categorie': row['categorie'],
                'type_question': 'usage_medicament',
                'necessite_avertissement': True,
            })
        
        return paires
    
    def generer_paires_procedure(self, row: pd.Series) -> List[Dict]:
        """Génère des paires Q&A pour les procédures"""
        paires = []
        
        if row['type'] != 'procedure':
            return paires
        
        questions = self.choisir_templates(TEMPLATES_QUESTIONS['procedure_deroulement'], 1)
        reponse = f"{row['terme']} est une procédure médicale"
        
        if pd.notna(row['categorie']):
            reponse += f" {row['categorie']}"
        
        reponse += ". L'équipe médicale vous expliquera le déroulement détaillé avant l'intervention."
        
        for question in questions:
            paires.append({
                'question': question.format(terme=row['terme']),
                'reponse': reponse,
                'terme_original': row['terme'],
                'type': row['type'],
                'categorie': row['categorie'],
                'type_question': 'procedure',
                'necessite_avertissement': False,
            })
        
        return paires
    
    def ajouter_avertissement(self, paire: Dict) -> Dict:
        """Ajoute un avertissement si nécessaire"""
        if not paire['necessite_avertissement']:
            return paire
        
        type_q = paire['type_question']
        
        if type_q in ['interpretation', 'valeurs_normales', 'valeurs_critiques']:
            paire['reponse'] += f"\n\n{AVERTISSEMENTS['interpretation']}"
        elif type_q in ['symptomes']:
            paire['reponse'] += f"\n\n{AVERTISSEMENTS['diagnostic']}"
        elif type_q in ['traitement', 'usage_medicament']:
            paire['reponse'] += f"\n\n{AVERTISSEMENTS['traitement']}"
        
        self.stats['avertissements'] += 1
        return paire
    
    def augmenter_ligne(self, row: pd.Series) -> List[Dict]:
        """Augmente une ligne du dataset"""
        paires = []
        
        # Générer toutes les paires possibles
        paires.extend(self.generer_paires_definition(row))
        paires.extend(self.generer_paires_abbreviation(row))
        paires.extend(self.generer_paires_valeurs_normales(row))
        paires.extend(self.generer_paires_valeurs_critiques(row))
        paires.extend(self.generer_paires_interpretation(row))
        paires.extend(self.generer_paires_symptomes(row))
        paires.extend(self.generer_paires_traitement(row))
        paires.extend(self.generer_paires_causes(row))
        paires.extend(self.generer_paires_medicament(row))
        paires.extend(self.generer_paires_procedure(row))
        
        # Ajouter les avertissements
        paires = [self.ajouter_avertissement(p) for p in paires]
        
        # Limiter au nombre souhaité
        if len(paires) > self.variations_par_terme:
            paires = random.sample(paires, self.variations_par_terme)
        
        return paires
    
    def augmenter_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Augmente tout le dataset"""
        print("\n" + "=" * 80)
        print("AUGMENTATION DES DONNÉES")
        print("=" * 80)
        
        donnees = []
        
        for idx, row in df.iterrows():
            paires = self.augmenter_ligne(row)
            donnees.extend(paires)
            
            # Stats
            self.stats['total_paires'] += len(paires)
            self.stats['par_type'][row['type']] = self.stats['par_type'].get(row['type'], 0) + len(paires)
            
            for paire in paires:
                type_q = paire['type_question']
                self.stats['par_type_question'][type_q] = self.stats['par_type_question'].get(type_q, 0) + 1
            
            if (idx + 1) % 25 == 0:
                print(f"   Traité: {idx + 1}/{len(df)} termes ({len(donnees)} paires générées)")
        
        print(f"\n✅ Augmentation terminée: {len(donnees)} paires Q&A")
        print(f"   Avertissements ajoutés: {self.stats['avertissements']}")
        
        return pd.DataFrame(donnees)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def charger_dataset(chemin: str) -> pd.DataFrame:
    """Charge le dataset"""
    print("=" * 80)
    print("CHARGEMENT DU DATASET")
    print("=" * 80)
    
    try:
        df = pd.read_csv(chemin, encoding='utf-8-sig')
        print(f"\n✓ Dataset chargé: {len(df)} entrées")
        print(f"\n  Distribution par type:")
        for type_val, count in df['type'].value_counts().items():
            print(f"    • {type_val}: {count}")
        return df
    except FileNotFoundError:
        print(f"\n❌ Fichier introuvable: {chemin}")
        return None
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

def sauvegarder_resultats(df_augmente: pd.DataFrame, augmentateur: AugmentateurMedical):
    """Sauvegarde les résultats"""
    print("\n" + "=" * 80)
    print("SAUVEGARDE DES RÉSULTATS")
    print("=" * 80)
    
    dossier = Path(DOSSIER_SORTIE)
    dossier.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV principal
    chemin_csv = dossier / "medical_qa_augmented.csv"
    df_augmente.to_csv(chemin_csv, index=False, encoding='utf-8-sig')
    print(f"\n✓ CSV: {chemin_csv}")
    
    # CSV avec timestamp
    chemin_csv_ts = dossier / f"medical_qa_{timestamp}.csv"
    df_augmente.to_csv(chemin_csv_ts, index=False, encoding='utf-8-sig')
    print(f"✓ CSV timestampé: {chemin_csv_ts}")
    
    # Excel avec stats
    try:
        chemin_excel = dossier / "medical_qa_augmented.xlsx"
        with pd.ExcelWriter(chemin_excel, engine='openpyxl') as writer:
            df_augmente.to_excel(writer, sheet_name='Q&A', index=False)
            
            # Stats
            stats_type = df_augmente.groupby('type').size().reset_index(name='count')
            stats_type.to_excel(writer, sheet_name='Stats Type', index=False)
            
            stats_question = df_augmente.groupby('type_question').size().reset_index(name='count')
            stats_question.to_excel(writer, sheet_name='Stats Question', index=False)
        
        print(f"✓ Excel: {chemin_excel}")
    except Exception as e:
        print(f"⚠️  Excel non sauvegardé: {e}")
    
    # Rapport
    chemin_rapport = dossier / "rapport.txt"
    with open(chemin_rapport, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RAPPORT D'AUGMENTATION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date: {datetime.now()}\n\n")
        f.write(f"Total paires: {len(df_augmente)}\n")
        f.write(f"Avertissements: {augmentateur.stats['avertissements']}\n\n")
        
        f.write("PAR TYPE:\n")
        for type_val, count in sorted(augmentateur.stats['par_type'].items()):
            f.write(f"  {type_val}: {count}\n")
        
        f.write("\nPAR TYPE DE QUESTION:\n")
        for type_q, count in sorted(augmentateur.stats['par_type_question'].items()):
            f.write(f"  {type_q}: {count}\n")
    
    print(f"✓ Rapport: {chemin_rapport}")
    print(f"\n📁 Dossier: {dossier}")

def afficher_exemples(df: pd.DataFrame):
    """Affiche des exemples"""
    print("\n" + "=" * 80)
    print("EXEMPLES")
    print("=" * 80)
    
    for type_ex in df['type'].unique()[:3]:
        subset = df[df['type'] == type_ex]
        if not subset.empty:
            ex = subset.sample(1).iloc[0]
            print(f"\n📝 {type_ex.upper()}")
            print(f"   Question: {ex['question']}")
            print(f"   Réponse: {ex['reponse'][:150]}...")

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "DATA AUGMENTATION Q&A MÉDICAL - VERSION COMPLÈTE" + " " * 14 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Charger
    df = charger_dataset(CHEMIN_ENTREE)
    if df is None:
        return
    
    # Augmenter
    augmentateur = AugmentateurMedical(variations_par_terme=VARIATIONS_PAR_TERME)
    df_augmente = augmentateur.augmenter_dataset(df)
    
    # Sauvegarder
    sauvegarder_resultats(df_augmente, augmentateur)
    
    # Exemples
    afficher_exemples(df_augmente)
    
    # Final
    print("\n" + "=" * 80)
    print("✅ TERMINÉ AVEC SUCCÈS!")
    print("=" * 80)
    print(f"\n📊 Résumé:")
    print(f"   • {len(df_augmente)} paires Q&A")
    print(f"   • {augmentateur.stats['avertissements']} avertissements médicaux")
    print(f"   • Ratio: ~{len(df_augmente)//len(df)} paires/terme")
    print(f"\n💡 Fichier prêt: medical_qa_augmented.csv")
    print(f"📁 Dossier: {DOSSIER_SORTIE}")
    print("\n⚠️  IMPORTANT: Validation médicale requise avant usage!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Arrêté par l'utilisateur.")
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()