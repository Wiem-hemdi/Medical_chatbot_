import gradio as gr
import subprocess
import re
from rapidfuzz import process
import requests
import json
import hashlib
import os
from datetime import datetime
import tempfile

print("="*60)
print("🏥 ASSISTANT MÉDICAL - PRIORITÉ OPTIMISÉE + PDF")
print("="*60)

# -----------------------------
# Import des modules personnalisés
# -----------------------------
try:
    from data_loader import data_loader
    print("✅ Module data_loader importé")
except ImportError as e:
    print(f"❌ Erreur d'import data_loader: {e}")
    print("Création d'un data_loader minimal...")
    class MinimalDataLoader:
        def __init__(self):
            self.terms_dict = {}
            self.qa_pairs = []
            self.dfs = {}
    data_loader = MinimalDataLoader()

try:
    from lab_utils import extract_lab_results_from_pdf, explain_input, search_medical_info, generate_pdf_report, get_normal_range
    print("✅ Module lab_utils importé")
except ImportError as e:
    print(f"❌ Erreur d'import lab_utils: {e}")
    def explain_input(text):
        return f"<div class='warning-box'>Module lab_utils non disponible</div>"
    def search_medical_info(query):
        return None
    def get_normal_range(test_name):
        return None

try:
    from rag_utils import rag_system, build_rag_from_loader, get_rag_context
    print("✅ Module rag_utils importé")
    
    # Construction du système RAG
    print("🧠 Construction du système RAG...")
    try:
        build_rag_from_loader(data_loader)
        print("✅ Système RAG construit avec succès")
    except Exception as e:
        print(f"⚠️ Erreur construction RAG: {e}")
        rag_system = None
        get_rag_context = lambda x, top_k=3: None
except ImportError as e:
    print(f"❌ Erreur d'import rag_utils: {e}")
    rag_system = None
    get_rag_context = lambda x, top_k=3: None

# -----------------------------
# Configuration
# -----------------------------
OLLAMA_PATH = "ollama"
MODEL_NAME = "qwen2.5:0.5b-instruct-q4_K_M"
WIKI_CACHE_FILE = "wiki_cache.json"

# -----------------------------
# Wikipedia Cache
# -----------------------------
class WikiCache:
    def __init__(self, cache_file=WIKI_CACHE_FILE):
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self):
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except:
            pass
        return {}
    
    def _save_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except:
            pass
    
    def get(self, term):
        key = hashlib.md5(term.strip().lower().encode()).hexdigest()
        return self.cache.get(key)
    
    def set(self, term, content):
        key = hashlib.md5(term.strip().lower().encode()).hexdigest()
        self.cache[key] = content
        if len(self.cache) > 200:
            keys = list(self.cache.keys())
            for k in keys[:50]:
                del self.cache[k]
        self._save_cache()

wiki_cache = WikiCache()

# -----------------------------
# Wikipedia Function 
# -----------------------------
def get_wikipedia_summary(term, lang='fr'):
    """Utilisé seulement si toutes les autres méthodes échouent"""
    try:
        cached = wiki_cache.get(term)
        if cached:
            return cached
        
        # Filtrer les termes non pertinents
        excluded_words = {'comment', 'pourquoi', 'quand', 'où', 'combien', 
                         'quel', 'quelle', 'quels', 'quelles', 'est-ce'}
        
        # Nettoyer le terme
        clean_term = term.strip().lower()
        words = clean_term.split()
        
        # Si c'est une question, extraire le sujet
        if any(q_word in clean_term for q_word in ['?', 'quoi', 'comment', 'pourquoi']):
            # Extraire les mots potentiellement médicaux
            medical_keywords = ['hémoglobine', 'cholestérol', 'diabète', 'hypertension', 
                              'créatinine', 'glycémie', 'insuline', 'thyroïde', 'cardiaque']
            for word in words:
                if word in medical_keywords:
                    clean_term = word
                    break
            else:
                # Prendre le mot le plus long qui n'est pas un mot interrogatif
                for word in sorted(words, key=len, reverse=True):
                    if len(word) > 4 and word not in excluded_words:
                        clean_term = word
                        break
        
        if len(clean_term) < 3:
            return None
        
        search_term = clean_term.replace(' ', '_')
        
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{search_term}"
        
        headers = {
            'User-Agent': 'MedicalBot/1.0',
            'Accept': 'application/json'
        }
        
        response = requests.get(url, headers=headers, timeout=8)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'extract' in data and data['extract']:
                summary = data['extract']
                summary = re.sub(r'\s+', ' ', summary)
                summary = summary[:500] + "..." if len(summary) > 500 else summary
                
                title = data.get('title', clean_term.title())
                formatted = f"**📚 Wikipedia - {title}**\n\n{summary}"
                
                wiki_cache.set(term, formatted)
                return formatted
        
        return None
        
    except:
        return None

def analyze_pdf_file(pdf_file):
    """
    Analyse un fichier PDF uploadé
    """
    try:
        if pdf_file is None:
            return "⚠️ Aucun fichier sélectionné"
        
        print(f"📤 Analyse du fichier PDF: {pdf_file.name}")
        
        # Sauvegarder le fichier temporairement
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            # Lire le contenu du fichier uploadé
            with open(pdf_file.name, 'rb') as f:
                tmp_file.write(f.read())
            tmp_path = tmp_file.name
        
        try:
            # Analyser le PDF
            pdf_analysis = extract_lab_results_from_pdf(tmp_path)
            
            # Générer le rapport
            report = generate_pdf_report(pdf_analysis)
            
            return report
            
        finally:
            # Nettoyer le fichier temporaire
            try:
                os.unlink(tmp_path)
            except:
                pass
                
    except Exception as e:
        print(f"❌ Erreur analyse PDF: {e}")
        return f"""
        <div class="warning-box">
            <h4>❌ Erreur d'analyse</h4>
            <p>Impossible d'analyser le fichier PDF: {str(e)}</p>
            <p><small>Assurez-vous qu'il s'agit d'un PDF valide contenant des résultats de laboratoire.</small></p>
        </div>
        """
# -----------------------------
# IA Function 
# -----------------------------
def ask_phi(question, context=None):
    """IA standard - 4ème priorité"""
    try:
        prompt = f"""Tu es un assistant médical francophone expert.

{'[CONTEXTE SUPPLÉMENTAIRE] ' + context if context else ''}

Question: {question}

Instructions:
1. Réponds en français simple et clair
2. Sois concis (2-4 phrases maximum)
3. Si tu ne sais pas, recommande de consulter un médecin
4. Ne donne pas de diagnostic personnel

Réponse:"""
        
        result = subprocess.run(
            [OLLAMA_PATH, "run", MODEL_NAME],
            input=prompt,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=12,
            shell=False
        )
        
        if result.returncode == 0 and result.stdout:
            response = result.stdout.strip()
            response = re.sub(r'\n\s*\n+', '\n\n', response)
            return response
        
        return None
        
    except Exception as e:
        print(f"Erreur IA: {e}")
        return None

# -----------------------------
# RAG Function 
# -----------------------------
def get_rag_response(query):
    """Recherche avec RAG - 2ème priorité"""
    if not rag_system:
        return None
    
    try:
        context = get_rag_context(query, top_k=2)
        if not context:
            return None
        
        # Formater la réponse RAG
        lines = context.split('\n')
        formatted_response = []
        
        for line in lines:
            if line.strip() and not line.startswith('['):
                formatted_response.append(line.strip())
        
        if formatted_response:
            return {
                'content': '\n'.join(formatted_response[:3]), 
                'source': 'RAG',
                'has_context': True
            }
        
        return None
        
    except Exception as e:
        print(f"Erreur RAG: {e}")
        return None

# -----------------------------
# Dataset Direct Search 
# -----------------------------
def search_dataset_direct(query):
    """Recherche directe dans le dataset - 3ème priorité"""
    if not hasattr(data_loader, 'search_term'):
        return None
    
    # Chercher le terme exact
    term_data = data_loader.search_term(query)
    if term_data:
        explanation = term_data.get('explanation', '')
        if explanation:
            return {
                'content': explanation,
                'source': term_data.get('source', 'dataset'),
                'term': term_data.get('row_data', {}).get('term', query)
            }
    
    # Chercher dans les QA
    if hasattr(data_loader, 'search_qa'):
        qa_answer = data_loader.search_qa(query)
        if qa_answer:
            return {
                'content': qa_answer,
                'source': 'qa_dataset',
                'is_qa': True
            }
    
    return None

# -----------------------------
# Main Processing Function 
# -----------------------------
def process_medical_query(user_input):
    """
    Traite les requêtes médicales avec la nouvelle priorité:
    1. Labo → 2. RAG → 3. Dataset → 4. IA → 5. Wikipedia
    """
    if not user_input or not user_input.strip():
        return """<div style="text-align: center; padding: 40px; color: #666;">
                    <h3>👋 Assistant Médical</h3>
                    <p>Posez votre question ou entrez vos résultats.</p>
                    <p><small>Exemples: "Hémoglobine 14" ou "Qu'est-ce que le diabète?"</small></p>
                 </div>"""
    
    user_lower = user_input.lower().strip()
    print(f"\n[QUESTION] '{user_input}'")
    
    # ==================== 1. PRIORITÉ: RÉSULTATS DE LABORATOIRE ====================
    print("   🔬 Vérification résultats labo...")
    
    # Détection améliorée des résultats labo
    lab_keywords = ['hémoglobine', 'hb', 'cholestérol', 'chol', 'ldl', 'hdl', 
                   'créatinine', 'cr', 'glycémie', 'glucose', 'tsh', 't4', 'ft4',
                   'alt', 'ast', 'pal', 'sodium', 'na', 'potassium', 'k', 'urée',
                   'bun', 'crp', 'vs', 'plaquettes', 'plt', 'leucocytes', 'gb']
    
    has_lab_keyword = any(keyword in user_lower for keyword in lab_keywords)
    has_numbers = any(char.isdigit() for char in user_input)
    
    if has_lab_keyword and has_numbers:
        print("   🧪 Détection labo positive")
        lab_response = explain_input(user_input)
        
        # Vérifier si c'est une vraie réponse labo
        if lab_response and "Aucun résultat" not in lab_response and "Erreur" not in lab_response:
            print("   ✅ Réponse labo générée")
            
            # Ajouter un conseil si valeur anormale
            if "élevé" in lab_response.lower() or "bas" in lab_response.lower():
                lab_response += """
                <div style="margin-top: 15px; padding: 12px; background: #fff3cd; border-radius: 6px; border-left: 3px solid #f39c12;">
                    <strong>💡 Conseil:</strong> Un résultat anormal nécessite une consultation médicale pour interprétation complète.
                </div>
                """
            
            return f"""
            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <div style="display: flex; align-items: center; margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px solid #eee;">
                    <div style="background: #3498db; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px; font-size: 20px;">
                        🧪
                    </div>
                    <div>
                        <h3 style="margin: 0; color: #2c3e50;">Analyse de résultat de laboratoire</h3>
                        <small style="color: #7f8c8d;">Interprétation automatique</small>
                    </div>
                </div>
                {lab_response}
            </div>
            """
    
    # ==================== 2. PRIORITÉ: RECHERCHE RAG ====================
    if rag_system:
        print("   🔍 Recherche RAG...")
        rag_response = get_rag_response(user_input)
        
        if rag_response:
            print(f"   ✅ RAG trouvé (source: {rag_response.get('source', 'RAG')})")
            
            source_badge = ""
            if rag_response.get('source'):
                source_badge = f"<span style='background: #9b59b6; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin-left: 10px;'>{rag_response['source']}</span>"
            
            response_html = f"""
            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <div style="display: flex; align-items: center; margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px solid #eee;">
                    <div style="background: #9b59b6; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px; font-size: 20px;">
                        🧠
                    </div>
                    <div>
                        <h3 style="margin: 0; color: #2c3e50;">Information trouvée par recherche intelligente {source_badge}</h3>
                        <small style="color: #7f8c8d;">Système de recherche contextuelle (RAG)</small>
                    </div>
                </div>
                <div style="font-size: 16px; line-height: 1.6; color: #2c3e50; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                    {rag_response['content']}
                </div>
            """
            
            # Si le RAG a du contexte, on peut ajouter une explication IA
            if rag_response.get('has_context') and '?' in user_input:
                print("   🤖 Ajout explication IA avec contexte RAG...")
                ai_extra = ask_phi(user_input, rag_response['content'])
                if ai_extra:
                    response_html += f"""
                    <div style="margin-top: 20px; padding: 15px; background: #e8f6f3; border-radius: 8px; border-left: 4px solid #1abc9c;">
                        <h4 style="margin-top: 0; color: #16a085;">💡 Explication complémentaire</h4>
                        {ai_extra}
                    </div>
                    """
            
            response_html += """
                <div style="margin-top: 20px; padding: 15px; background: #fff3cd; border-radius: 8px; border-left: 4px solid #ffc107;">
                    <strong>📚 Information de nos bases médicales</strong><br>
                    Cette réponse est extraite de nos données vérifiées.
                </div>
            </div>
            """
            
            return response_html
    
    # ==================== 3. PRIORITÉ: RECHERCHE DATASET DIRECT ====================
    print("   📊 Recherche dataset direct...")
    dataset_response = search_dataset_direct(user_input)
    
    if dataset_response:
        print(f"   ✅ Dataset trouvé (source: {dataset_response.get('source', 'dataset')})")
        
        source_color = "#3498db" if dataset_response.get('source') == 'dataset' else "#2ecc71"
        source_icon = "📖" if dataset_response.get('source') == 'dataset' else "💬"
        
        response_html = f"""
        <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px solid #eee;">
                <div style="background: {source_color}; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px; font-size: 20px;">
                    {source_icon}
                </div>
                <div>
                    <h3 style="margin: 0; color: #2c3e50;">{dataset_response.get('term', 'Information médicale')}</h3>
                    <small style="color: #7f8c8d;">Source: {dataset_response.get('source', 'Base de données')}</small>
                </div>
            </div>
            <div style="font-size: 16px; line-height: 1.6; color: #2c3e50;">
                {dataset_response['content']}
            </div>
        """
        
        # Si c'est une définition simple, ajouter des valeurs normales si disponibles
        if dataset_response.get('term') and not dataset_response.get('is_qa'):
            normal_range = get_normal_range(dataset_response['term'])
            if normal_range:
                response_html += f"""
                <div style="margin-top: 15px; padding: 12px; background: #e8f4fc; border-radius: 6px; border-left: 3px solid #3498db;">
                    <strong>📊 Valeurs de référence:</strong> {normal_range.get('min', '')} - {normal_range.get('max', '')} {normal_range.get('unit', '')}
                </div>
                """
        
        response_html += """
            <div style="margin-top: 20px; padding: 15px; background: #fff3cd; border-radius: 8px; border-left: 4px solid #ffc107;">
                <strong>⚠️ Information éducative</strong><br>
                Consultez un professionnel de santé pour un avis personnel.
            </div>
        </div>
        """
        
        return response_html
    
    # ==================== 4. PRIORITÉ: IA STANDARD ====================
    print("   🤖 Consultation IA standard...")
    ai_response = ask_phi(user_input)
    
    if ai_response and len(ai_response) > 20:  # Éviter les réponses trop courtes
        print("   ✅ Réponse IA générée")
        
        return f"""
        <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px solid #eee;">
                <div style="background: linear-gradient(135deg, #9b59b6, #8e44ad); color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px; font-size: 20px;">
                    🤖
                </div>
                <div>
                    <h3 style="margin: 0; color: #2c3e50;">Réponse de l'assistant IA</h3>
                    <small style="color: #7f8c8d;">Modèle médical local - {MODEL_NAME}</small>
                </div>
            </div>
            <div style="font-size: 16px; line-height: 1.6; color: #2c3e50; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                {ai_response}
            </div>
            <div style="margin-top: 20px; padding: 15px; background: #fff3cd; border-radius: 8px; border-left: 4px solid #ffc107;">
                <strong>⚠️ Réponse générée par IA</strong><br>
                Cette réponse est produite automatiquement et nécessite vérification par un professionnel.
            </div>
        </div>
        """
    
    # ==================== 5. PRIORITÉ: WIKIPEDIA (DERNIER RECOURS) ====================
    print("   🌐 Recherche Wikipedia (dernier recours)...")
    
    # Extraire les termes potentiels pour Wikipedia
    potential_terms = []
    
    # 1. Chercher des termes médicaux connus
    if hasattr(data_loader, 'terms_dict'):
        for term in data_loader.terms_dict.keys():
            if term in user_lower and len(term) > 4:
                potential_terms.append(term)
    
    # 2. Extraire les mots longs de la question
    words = user_lower.split()
    for word in words:
        if len(word) > 5 and word not in ['pourquoi', 'comment', 'quand', 'quelle']:
            potential_terms.append(word)
    
    # 3. Chercher sur Wikipedia
    for term in potential_terms[:3]:  # Limiter à 3 tentatives
        wiki_response = get_wikipedia_summary(term)
        if wiki_response:
            print(f"   ✅ Wikipedia trouvé pour '{term}'")
            
            return f"""
            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <div style="display: flex; align-items: center; margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px solid #eee;">
                    <div style="background: #e67e22; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px; font-size: 20px;">
                        🌐
                    </div>
                    <div>
                        <h3 style="margin: 0; color: #2c3e50;">Information générale</h3>
                        <small style="color: #7f8c8d;">Source: Wikipedia</small>
                    </div>
                </div>
                <div style="font-size: 16px; line-height: 1.6; color: #2c3e50;">
                    {wiki_response}
                </div>
                <div style="margin-top: 20px; padding: 15px; background: #fff3cd; border-radius: 8px; border-left: 4px solid #ffc107;">
                    <strong>⚠️ Source encyclopédique</strong><br>
                    Cette information provient de Wikipedia et est à titre informatif seulement.
                </div>
            </div>
            """
    
    # ==================== AUCUNE RÉPONSE ====================
    print("   ❌ Aucune information trouvée")
    
    # Suggestions contextuelles
    suggestions = []
    
    # Si ça ressemble à une question labo mal formulée
    if has_numbers:
        suggestions.append("Hémoglobine 14 g/dL (analyse labo)")
        suggestions.append("Glucose 5.6 mmol/L (glycémie)")
    
    # Si c'est une question générale
    elif '?' in user_input:
        suggestions.append("Qu'est-ce que l'hémoglobine? (définition)")
        suggestions.append("Valeurs normales du cholestérol (références)")
    
    # Suggestions par défaut
    if not suggestions:
        suggestions = [
            "Hémoglobine 14.5 g/dL (analyse labo)",
            "Qu'est-ce que le diabète de type 2? (définition)",
            "Symptômes de l'hypertension (informations symptômes)",
            "Valeurs normales TSH (références médicales)"
        ]
    
    # Créer les suggestions avec JavaScript propre
    suggestions_html = ""
    for i, suggestion in enumerate(suggestions, 1):
        # Extraire le texte à insérer (avant la parenthèse)
        insert_text = suggestion.split('(')[0].strip()
        # Échapper les guillemets simples pour JavaScript
        js_text = insert_text.replace("'", "\\'").replace('"', '\\"')
        
        suggestions_html += f"""
        <div class="suggestion-item" 
             onclick="setExample('{js_text}')"
             style="margin: 8px 0; padding: 12px; background: #f8f9fa; border-radius: 6px; border-left: 3px solid #3498db; cursor: pointer; transition: all 0.2s;">
            <div style="font-weight: 500; color: #2c3e50;">{suggestion}</div>
            <div style="font-size: 12px; color: #7f8c8d; margin-top: 4px;">Cliquez pour essayer</div>
        </div>
        """
    
    return f"""
    <div style="background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center;">
        <div style="font-size: 48px; color: #95a5a6; margin-bottom: 20px;">🔍</div>
        <h3 style="color: #2c3e50; margin-bottom: 10px;">Je n'ai pas trouvé d'information spécifique</h3>
        <p style="color: #7f8c8d; margin-bottom: 25px;">
            Essayez une de ces formulations pour de meilleurs résultats:
        </p>
        
        <div style="max-width: 600px; margin: 0 auto;" id="suggestions-container">
            {suggestions_html}
        </div>
        
        <div style="margin-top: 30px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
            <p style="margin: 0; color: #7f8c8d; font-size: 14px;">
                <strong>💡 Conseil:</strong> Pour les résultats de laboratoire, incluez toujours la valeur et l'unité.<br>
                <strong>📊 Statistiques:</strong> Base de données: {len(data_loader.terms_dict) if hasattr(data_loader, 'terms_dict') else '?'} termes médicaux
            </p>
        </div>
        
        <script>
        function setExample(text) {{
            // Trouver la zone de texte de Gradio
            const textareas = document.querySelectorAll('textarea');
            if (textareas.length > 0) {{
                // Sélectionner la première zone de texte (votre input principal)
                textareas[0].value = text;
                // Déclencher l'événement input pour que Gradio détecte le changement
                textareas[0].dispatchEvent(new Event('input', {{ bubbles: true }}));
                // Optionnel: mettre le focus
                textareas[0].focus();
            }}
        }}
        
        // Ajouter des effets hover
        document.querySelectorAll('.suggestion-item').forEach(item => {{
            item.addEventListener('mouseenter', function() {{
                this.style.transform = 'translateY(-2px)';
                this.style.boxShadow = '0 4px 8px rgba(0,0,0,0.1)';
                this.style.background = '#e8f4fc';
            }});
            item.addEventListener('mouseleave', function() {{
                this.style.transform = 'none';
                this.style.boxShadow = 'none';
                this.style.background = '#f8f9fa';
            }});
        }});
        </script>
    </div>
    """
# -----------------------------
# Gradio Interface
# -----------------------------
def create_interface():
    """Crée l'interface utilisateur"""
    
    css = """
    .lab-result {
        padding: 20px;
        margin: 15px 0;
        border-radius: 10px;
        border-left: 5px solid;
        background: white;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
    }
    .normal-value { border-left-color: #27ae60 !important; background: linear-gradient(90deg, rgba(39, 174, 96, 0.05), white) !important; }
    .low-value { border-left-color: #3498db !important; background: linear-gradient(90deg, rgba(52, 152, 219, 0.05), white) !important; }
    .high-value { border-left-color: #e67e22 !important; background: linear-gradient(90deg, rgba(230, 126, 34, 0.05), white) !important; }
    .rag-value { border-left-color: #9b59b6 !important; background: linear-gradient(90deg, rgba(155, 89, 182, 0.05), white) !important; }
    .warning-box {
        padding: 15px;
        background: #fff3cd;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 15px 0;
    }
    .priority-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
        margin-left: 8px;
    }
    .priority-1 { background: #3498db; color: white; }
    .priority-2 { background: #9b59b6; color: white; }
    .priority-3 { background: #2ecc71; color: white; }
    .priority-4 { background: #f39c12; color: white; }
    .priority-5 { background: #e67e22; color: white; }
    .pdf-upload-area {
        border: 2px dashed #3498db;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background: #f8f9fa;
        transition: all 0.3s;
    }
    .pdf-upload-area:hover {
        background: #e8f4fc;
        border-color: #2980b9;
    }
    .test-result {
        transition: all 0.3s ease;
    }
    .test-result:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    """
    
    with gr.Blocks(title="Assistant Médical - Priorité: Labo > RAG > Dataset > IA > Wikipedia", 
                   css=css, theme=gr.themes.Soft()) as demo:
        
        # Header
        rag_status = "✅ Activé" if rag_system else "❌ Désactivé"
        
        gr.Markdown(f"""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="margin-bottom: 15px; color: #2c3e50;">🏥 Assistant Médical Intelligent</h1>
            <div style="display: inline-flex; flex-wrap: wrap; gap: 10px; justify-content: center; margin-bottom: 20px;">
                <div style="background: #e8f4fc; padding: 6px 12px; border-radius: 20px; font-size: 14px;">
                    <strong>Priorité:</strong> Labo → RAG → Dataset → IA → Wikipedia
                </div>
                <div style="background: #fdebd0; padding: 6px 12px; border-radius: 20px; font-size: 14px;">
                    <strong>RAG:</strong> {rag_status}
                </div>
                <div style="background: #e8f6f3; padding: 6px 12px; border-radius: 20px; font-size: 14px;">
                    <strong>IA:</strong> {MODEL_NAME}
                </div>
                <div style="background: #e8f4fc; padding: 6px 12px; border-radius: 20px; font-size: 14px;">
                    <strong>PDF:</strong> ✅ Supporté
                </div>
            </div>
        </div>
        """)
        
        # Zone de saisie
        with gr.Row():
            with gr.Column(scale=3):
                user_input = gr.Textbox(
                    label="💬 Votre question ou résultat médical",
                    placeholder="Exemples:\n• Hémoglobine 14 g/dL (analyse labo)\n• Qu'est-ce que le cholestérol LDL? (question)\n• TSH 2.5 mUI/L (résultat)",
                    lines=3
                )
                
                with gr.Row():
                    submit_btn = gr.Button("🚀 Analyser", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Effacer", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown(f"""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 10px;">
                    <h4 style="margin-top: 0; margin-bottom: 15px;">🎯 Ordre de priorité</h4>
                    <div style="margin: 8px 0;">
                        <span class="priority-badge priority-1">1</span> Analyse résultats labo
                    </div>
                    <div style="margin: 8px 0;">
                        <span class="priority-badge priority-2">2</span> Recherche RAG
                    </div>
                    <div style="margin: 8px 0;">
                        <span class="priority-badge priority-3">3</span> Base de données
                    </div>
                    <div style="margin: 8px 0;">
                        <span class="priority-badge priority-4">4</span> IA locale
                    </div>
                    <div style="margin: 8px 0;">
                        <span class="priority-badge priority-5">5</span> Wikipedia
                    </div>
                    <div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #dee2e6;">
                        <small style="color: #6c757d;">📄 PDF: Analyse séparée</small>
                    </div>
                </div>
                """)
        
        # Zone de résultat
        output_area = gr.HTML(
            label="📋 Résultat",
            value="""<div style="text-align: center; padding: 40px; color: #666;">
                    <h3>👋 Prêt à analyser</h3>
                    <p>Le système utilisera l'ordre de priorité configuré.</p>
                    <p><small>Essayez un des exemples ci-dessous</small></p>
                 </div>"""
        )
        
        # Exemples
        gr.Markdown("### 💡 Exemples de test")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 🧪 **Résultats labo** <span class='priority-badge priority-1'>1</span>")
                gr.Examples(
                    examples=[
                        ["Hémoglobine 14.5 g/dL"],
                        ["Cholestérol total 5.8 mmol/L"],
                        ["Créatinine 85 μmol/L"],
                        ["TSH 2.5 mUI/L"]
                    ],
                    inputs=user_input,
                    outputs=output_area,
                    fn=process_medical_query
                )
            
            with gr.Column():
                gr.Markdown("#### 🧠 **Questions RAG** <span class='priority-badge priority-2'>2</span>")
                gr.Examples(
                    examples=[
                        ["Qu'est-ce que l'hémoglobine glyquée?"],
                        ["Fonction de la créatinine"],
                        ["Différence entre HDL et LDL"],
                        ["Rôle de la TSH"]
                    ],
                    inputs=user_input,
                    outputs=output_area,
                    fn=process_medical_query
                )
            
            with gr.Column():
                gr.Markdown("#### 🤖 **Questions IA** <span class='priority-badge priority-4'>4</span>")
                gr.Examples(
                    examples=[
                        ["Comment prévenir le diabète?"],
                        ["Quels sont les symptômes de l'hypothyroïdie?"],
                        ["Quand faut-il consulter pour de l'hypertension?"],
                        ["Comment fonctionne la metformine?"]
                    ],
                    inputs=user_input,
                    outputs=output_area,
                    fn=process_medical_query
                )
        
        # ==================== SECTION UPLOAD PDF ====================
        gr.Markdown("---")
        gr.Markdown("### 📄 **Analyse de compte-rendu PDF**")
        
        with gr.Row():
            with gr.Column(scale=2):
                pdf_upload = gr.File(
                    label="Téléchargez votre compte-rendu de laboratoire (PDF)",
                    file_types=[".pdf"],
                    type="filepath"
                )
                
                with gr.Row():
                    analyze_pdf_btn = gr.Button("🔬 Analyser le PDF", variant="secondary")
                    clear_pdf_btn = gr.Button("🗑️ Effacer")
            
            with gr.Column(scale=1):
                gr.Markdown("""
                <div style="background: #f0f8ff; padding: 15px; border-radius: 10px;">
                    <h4 style="margin-top: 0;">📋 Formats supportés</h4>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        <li>Comptes-rendus de laboratoire</li>
                        <li>Résultats d'analyses médicales</li>
                        <li>Bilans sanguins</li>
                        <li>Format PDF standard</li>
                    </ul>
                    <p><small>Le système extrait et interprète automatiquement les résultats.</small></p>
                </div>
                """)
        
        pdf_output = gr.HTML(
            label="📊 Rapport d'analyse",
            value="<div style='text-align: center; padding: 40px; color: #666;'>Votre rapport apparaîtra ici après analyse du PDF.</div>"
        )
        
        # ==================== HANDLERS PDF ====================
        analyze_pdf_btn.click(
            fn=analyze_pdf_file,
            inputs=pdf_upload,
            outputs=pdf_output
        )
        
        clear_pdf_btn.click(
            fn=lambda: [None, "<div style='text-align: center; padding: 40px; color: #666;'>Prêt pour l'analyse PDF.</div>"],
            inputs=[],
            outputs=[pdf_upload, pdf_output]
        )
        
        # ==================== INFORMATIONS SYSTÈME ====================
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 📊 **Système**")
                gr.Markdown(f"""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                    <p><strong>Capacités:</strong></p>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        <li>Interprétation résultats labo</li>
                        <li>Recherche contextuelle (RAG)</li>
                        <li>Réponses IA locales</li>
                        <li>Informations Wikipedia</li>
                        <li>Analyse PDF de résultats</li>
                    </ul>
                    <p><strong>Données:</strong></p>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        <li>Dataset original médical</li>
                        <li>2 datasets augmentés</li>
                        <li>Total: {len(data_loader.terms_dict) if hasattr(data_loader, 'terms_dict') else '?'} termes</li>
                    </ul>
                </div>
                """)
            
            with gr.Column():
                gr.Markdown("#### 🚨 **Urgences**")
                gr.HTML("""
                <div style="text-align: center;">
                    <a href="tel:190" style="text-decoration: none;">
                        <div style="background: #e74c3c; color: white; padding: 20px; border-radius: 10px; margin: 5px; transition: transform 0.2s;">
                            <div style="font-size: 28px;">🚑</div>
                            <div style="font-size: 24px; font-weight: bold;">190</div>
                            <div>SAMU</div>
                            <small>Urgences médicales</small>
                        </div>
                    </a>
                </div>
                """)
        
        # Disclaimer
        gr.Markdown("""
        <div style="padding: 20px; background: #f8d7da; border-radius: 10px; border-left: 5px solid #dc3545; margin-top: 20px;">
            <div style="display: flex; align-items: flex-start;">
                <div style="font-size: 24px; margin-right: 15px;">⚠️</div>
                <div>
                    <strong style="color: #721c24;">INFORMATION ÉDUCATIVE SEULEMENT - PROJET ACADÉMIQUE</strong><br>
                    <span style="color: #856404;">
                    • Cet assistant ne fournit pas de diagnostic médical<br>
                    • Les réponses sont générées à partir de sources variées<br>
                    • Consultez toujours un professionnel de santé pour tout problème médical<br>
                    • En cas d'urgence, appelez le 15 immédiatement
                    </span>
                </div>
            </div>
        </div>
        """)
        
        # ==================== HANDLERS PRINCIPAUX ====================
        # Actions principales
        def clear_all():
            return "", """<div style="text-align: center; padding: 40px; color: #666;">
                    <h3>👋 Assistant prêt</h3>
                    <p>Le système utilise la priorité: Labo → RAG → Dataset → IA → Wikipedia</p>
                 </div>"""
        
        submit_btn.click(
            fn=process_medical_query,
            inputs=user_input,
            outputs=output_area
        )
        
        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[user_input, output_area]
        )
        
        user_input.submit(
            fn=process_medical_query,
            inputs=user_input,
            outputs=output_area
        )
    
    return demo

# -----------------------------
# Main Entry Point
# -----------------------------
if __name__ == "__main__":
    print(f"\n✅ Configuration système:")
    print(f"   🧪 Détection labo: Activée")
    print(f"   🧠 Système RAG: {'Activé' if rag_system else 'Désactivé'}")
    print(f"   📊 Dataset: {len(data_loader.terms_dict) if hasattr(data_loader, 'terms_dict') else 'N/A'} termes")
    print(f"   🤖 IA: {MODEL_NAME}")
    print(f"   🌐 Wikipedia: Dernier recours")
    print(f"   📄 Analyse PDF: ✅ Activée")
    
    print(f"\n🎯 Ordre de priorité établi:")
    print("   1. 🔬 Résultats laboratoire")
    print("   2. 🧠 Recherche RAG")
    print("   3. 📊 Dataset direct")
    print("   4. 🤖 IA standard")
    print("   5. 🌐 Wikipedia")
    print("   📄 PDF: Analyse séparée (hors priorité)")
    
    print(f"\n🚀 Lancement sur le port 7862...")
    print(f"🌐 Ouvrez: http://localhost:7862")
    print("="*60)
    
    try:
        demo = create_interface()
        demo.launch(
            server_name="127.0.0.1",
            server_port=7862,
            share=False,
            inbrowser=True,
            quiet=False
        )
    except KeyboardInterrupt:
        print("\n👋 Arrêt de l'application...")
        wiki_cache._save_cache()
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()