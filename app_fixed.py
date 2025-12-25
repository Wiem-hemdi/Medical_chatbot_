# app_working_enhanced.py
import subprocess
import gradio as gr
import pandas as pd
import re
from rapidfuzz import process
import numpy as np

print("=" * 60)
print("🤖 Assistant Médical avec Phi-2.7b - VERSION AUGMENTÉE")
print("=" * 60)

# -----------------------------
# Configuration
# -----------------------------
OLLAMA_PATH = "ollama"
MODEL_NAME = "phi:2.7b"

# -----------------------------
# Chargement dataset ORIGINAL + AUGMENTATION MINIMALISTE
# -----------------------------
print("📊 Chargement et augmentation du dataset...")

def load_and_augment_dataset():
    """Charge le dataset original et ajoute des variantes minimales"""
    try:
        # Charger le dataset original
        df_original = pd.read_csv("datasets/comprehensive_medical_dataset.csv")
        print(f"✓ Dataset original chargé: {len(df_original)} termes")
        
        # Créer des variantes simples SANS modifier la structure
        augmented_rows = []
        
        for _, row in df_original.iterrows():
            # Ajouter la ligne originale
            augmented_rows.append(row.to_dict())
            
            # 1. Ajouter version française des termes anglais
            english_to_french = {
                "Hemoglobin": "Hémoglobine",
                "Glucose": "Glucose",
                "Cholesterol": "Cholestérol",
                "Creatinine": "Créatinine",
                "Diabetes": "Diabète",
                "Hypertension": "Hypertension"
            }
            
            if row['term'] in english_to_french:
                new_row = row.copy()
                new_row['term'] = english_to_french[row['term']]
                augmented_rows.append(new_row.to_dict())
            
            # 2. Ajouter abréviation comme terme principal si manquant
            if pd.notna(row['abbreviation']) and row['abbreviation'] != "":
                if row['abbreviation'] not in ["", "nan"]:
                    new_row = row.copy()
                    new_row['term'] = row['abbreviation']
                    new_row['abbreviation'] = row['term'][:3].lower() if len(row['term']) > 3 else row['term'].lower()
                    augmented_rows.append(new_row.to_dict())
        
        # Créer le DataFrame augmenté
        df_augmented = pd.DataFrame(augmented_rows)
        
        # Supprimer les doublons exacts
        df_augmented = df_augmented.drop_duplicates(subset=['term', 'explanation'], keep='first')
        
        print(f"✓ Dataset augmenté: {len(df_augmented)} termes (+{len(df_augmented)-len(df_original)})")
        return df_augmented
        
    except Exception as e:
        print(f"⚠️ Erreur dataset: {e}")
        print("Création d'un dataset minimal de test...")
        data = {
            'term': ['hémoglobine', 'glucose', 'cholestérol', 'créatinine', 'glycémie'],
            'abbreviation': ['hb', 'glu', 'chol', 'creat', ''],
            'normal_min': [12, 4, 1.5, 50, 4],
            'normal_max': [16, 6, 2.5, 110, 7],
            'unit': ['g/dl', 'mmol/l', 'g/l', 'μmol/l', 'mmol/l'],
            'explanation': [
                'Transporte l\'oxygène dans le sang. Valeur normale: 12-16 g/dl.',
                'Taux de sucre dans le sang. Valeur normale: 4-6 mmol/L.',
                'Graisse dans le sang. Valeur normale: 1.5-2.5 g/L.',
                'Indicateur de la fonction rénale. Valeur normale: 50-110 μmol/L.',
                'Taux de glucose à jeun. Valeur normale: 4-7 mmol/L.'
            ]
        }
        return pd.DataFrame(data)

# Charger le dataset augmenté
df = load_and_augment_dataset()

# Préparation des données (identique à votre code)
df['term'] = df['term'].str.lower()
df['abbreviation'] = df['abbreviation'].fillna("").str.lower()

# -----------------------------
# FONCTIONS EXACTEMENT COMME VOTRE CODE ORIGINAL
# -----------------------------
def ask_phi(question):
    """Pose une question au modèle Phi-2.7b"""
    try:
        # Prompt optimisé pour Phi-2.7b
        prompt = f"""Tu es un assistant médical francophone pour patients.

Question du patient: {question}

Instructions:
1. Réponds en français simple et clair
2. Utilise un langage accessible
3. Sois concis (3-4 phrases maximum)
4. Si tu ne sais pas, dis-le honnêtement
5. Recommande de consulter un médecin si nécessaire

Réponse:"""
        
        # Commande Ollama
        cmd = [OLLAMA_PATH, "run", MODEL_NAME]
        
        # Exécution
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=20,
            shell=True
        )
        
        if result.returncode != 0:
            print(f"⚠️ Erreur Ollama: {result.stderr}")
            return fallback_response(question)
        
        response = result.stdout.strip()
        
        if not response:
            return fallback_response(question)
        
        # Nettoyer la réponse
        response = re.sub(r'\n\s*\n+', '\n\n', response)
        
        return response
        
    except subprocess.TimeoutExpired:
        return "⏱️ La réponse prend trop de temps. Essayez une question plus courte."
    except Exception as e:
        print(f"⚠️ Exception: {str(e)}")
        return fallback_response(question)

def fallback_response(question):
    """Réponse de secours si Ollama échoue"""
    question_lower = question.lower()
    
    # Chercher dans le dataset AUGMENTÉ
    for _, row in df.iterrows():
        term_lower = row['term'].lower()
        abbrev = row['abbreviation'].lower() if pd.notna(row['abbreviation']) else ""
        
        if term_lower in question_lower or (abbrev and abbrev in question_lower):
            return f"**{row['term'].title()}**: {row.get('explanation', 'Information non disponible')}"
    
    return "Je n'ai pas d'information spécifique sur ce sujet. Pour une réponse précise, veuillez consulter un médecin."

def extract_lab_values(text):
    """Extrait les valeurs de laboratoire du texte"""
    text = text.lower()
    results = []
    
    # Pattern pour détecter "terme: valeur" ou "terme valeur"
    pattern = r'([a-zà-ÿ]+)\s*[:=]?\s*([0-9]+\.?[0-9]*)'
    matches = re.findall(pattern, text)
    
    for match in matches:
        term = match[0].strip()
        try:
            value = float(match[1])
            if len(term) > 2:
                results.append((term, value))
        except ValueError:
            continue
    
    return results

def interpret_lab_result(term, value):
    """Interprète un résultat de laboratoire"""
    # Chercher exact match dans le dataset AUGMENTÉ
    for _, row in df.iterrows():
        if term == row['term'] or term == row['abbreviation']:
            return format_interpretation(row, value)
    
    # Fuzzy matching sur dataset AUGMENTÉ
    all_terms = list(df['term']) + [abbr for abbr in df['abbreviation'] if abbr]
    best_match = process.extractOne(term, all_terms)
    
    if best_match and best_match[1] > 60:
        matched_term = best_match[0]
        for _, row in df.iterrows():
            if matched_term == row['term'] or matched_term == row['abbreviation']:
                return format_interpretation(row, value)
    
    return f"**{term.title()}**: {value} (terme non reconnu)"

def format_interpretation(row, value):
    """Formate l'interprétation d'un résultat"""
    try:
        if pd.notna(row.get('normal_min')) and pd.notna(row.get('normal_max')):
            min_val = float(row['normal_min'])
            max_val = float(row['normal_max'])
            unit = row.get('unit', '')
            
            if value < min_val:
                status = "🔵 BAS"
                color = "blue"
            elif value > max_val:
                status = "🟠 ÉLEVÉ"
                color = "orange"
            else:
                status = "🟢 NORMAL"
                color = "green"
            
            return f"**{row['term'].title()}**: {value} {unit}  \n{status} - Normal: {min_val}-{max_val} {unit}"
        else:
            return f"**{row['term'].title()}**: {value}  \n{row.get('explanation', '')}"
    except:
        return f"**{row['term'].title()}**: {value}  \n{row.get('explanation', '')}"

def process_medical_input(user_input):
    """Traite l'entrée utilisateur"""
    if not user_input.strip():
        return "Veuillez poser une question ou entrer vos résultats."
    
    output_parts = []
    
    # 1. Analyser les résultats de labo
    lab_results = extract_lab_values(user_input)
    
    if lab_results:
        output_parts.append("## 🔬 **Résultats détectés**")
        for term, value in lab_results:
            interpretation = interpret_lab_result(term, value)
            output_parts.append(f"- {interpretation}")
        output_parts.append("")
    
    # 2. Générer une réponse IA
    output_parts.append("## 💬 **Explication**")
    ai_response = ask_phi(user_input)
    output_parts.append(ai_response)
    
    # 3. Disclaimer
    output_parts.append("\n---")
    output_parts.append("⚠️ **Avertissement médical**: Cette information est générée par IA à titre éducatif seulement. **NE REMPLACE PAS** un avis médical professionnel. Consultez toujours un médecin.")
    
    return "\n".join(output_parts)

# -----------------------------
# Interface Gradio (identique)
# -----------------------------
def create_interface():
    """Crée l'interface utilisateur"""
    
    with gr.Blocks(title="Chatbot Médical") as demo:
        
        # Header amélioré avec info augmentation
        gr.Markdown(f"""
        # 🏥 Assistant Médical Intelligent
        ### Dataset: {len(df)} termes médicaux • Modèle: Phi-2.7b
        *Système optimisé avec reconnaissance avancée des termes médicaux*
        """)
        
        # Zone d'entrée
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="💬 Posez votre question médicale",
                    placeholder="""Exemples:
• 'Mes résultats: Hémoglobine 13.5, Glucose 5.8'
• 'Qu'est-ce que le cholestérol HDL?'
• 'Explique la différence entre infection virale et bactérienne'
• 'Crétatinine 120, est-ce normal?'""",
                    lines=6
                )
                
                with gr.Row():
                    submit_btn = gr.Button("🚀 Analyser", variant="primary")
                    clear_btn = gr.Button("🗑️ Effacer", variant="secondary")
        
        # Zone de sortie
        output_text = gr.Markdown(
            label="📋 Résultats et explications",
            value="Votre réponse apparaîtra ici..."
        )
        
        # Exemples enrichis
        gr.Markdown("### 💡 Exemples rapides")
        examples = gr.Examples(
            examples=[
                ["Hémoglobine 14.2, Glucose 6.1"],
                ["Qu'est-ce que la créatinine?"],
                ["Cholestérol total 2.4, Triglycérides 1.8"],
                ["Explique-moi ce qu'est le diabète"],
                ["Hb 15.0, WBC 8.5"]  # Nouveau avec abréviations
            ],
            inputs=input_text,
            outputs=output_text,
            fn=process_medical_input,
            cache_examples=False
        )
        
        # Info augmentation
        with gr.Accordion("ℹ️ Informations système", open=False):
            gr.Markdown(f"""
            **Système optimisé:**
            - 📊 **Dataset**: {len(df)} termes médicaux (augmenté)
            - 🤖 **Modèle IA**: Phi-2.7b pour réponses naturelles
            - 🔍 **Reconnaissance**: Termes + abréviations + synonymes
            - ⚡ **Performance**: Analyse automatique des résultats labo
            
            **Types reconnus:**
            - 🩸 Tests sanguins (Hémoglobine, Glucose, Cholestérol...)
            - 🏥 Maladies (Diabète, Hypertension...)
            - 📝 Abréviations (Hb, WBC, LDL, HDL...)
            - 🇫🇷 Termes français et anglais
            """)
        
        # Footer Tunisie
       # REMPLACER TOUT le footer actuel par :

        gr.Markdown("""
        ---
        ## 🚨 **Numéros d'urgence - Tunisie**

        <div style="
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0;
            color: white;
            text-align: center;
        ">
        <h3 style="color: white; margin-top: 0;">⚠️ En cas d'urgence médicale</h3>
        <p style="opacity: 0.9;">Cliquez directement sur les numéros pour appeler</p>
        </div>
        """)

        # Section des numéros d'urgence cliquables
        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div style="text-align: center; margin: 10px;">
                    <a href="tel:190" style="text-decoration: none;">
                        <div style="
                            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
                            color: white;
                            padding: 20px;
                            border-radius: 12px;
                            cursor: pointer;
                            font-size: 18px;
                            box-shadow: 0 6px 20px rgba(231, 76, 60, 0.3);
                            transition: all 0.3s ease;
                            margin: 10px;
                        " onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 12px 25px rgba(231, 76, 60, 0.4)';"
                        onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 6px 20px rgba(231, 76, 60, 0.3)';">
                            <div style="font-size: 30px; margin-bottom: 10px;">🚑</div>
                            <div style="font-size: 32px; font-weight: 800; margin: 10px 0;">190</div>
                            <div style="font-size: 16px; opacity: 0.9;">SAMU</div>
                            <div style="font-size: 14px; opacity: 0.8; margin-top: 5px;">Service d'Aide Médicale Urgente</div>
                        </div>
                    </a>
                </div>
                """)
            
            with gr.Column():
                gr.HTML("""
                <div style="text-align: center; margin: 10px;">
                    <a href="tel:197" style="text-decoration: none;">
                        <div style="
                            background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
                            color: white;
                            padding: 20px;
                            border-radius: 12px;
                            cursor: pointer;
                            font-size: 18px;
                            box-shadow: 0 6px 20px rgba(243, 156, 18, 0.3);
                            transition: all 0.3s ease;
                            margin: 10px;
                        " onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 12px 25px rgba(243, 156, 18, 0.4)';"
                        onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 6px 20px rgba(243, 156, 18, 0.3)';">
                            <div style="font-size: 30px; margin-bottom: 10px;">🚒</div>
                            <div style="font-size: 32px; font-weight: 800; margin: 10px 0;">197</div>
                            <div style="font-size: 16px; opacity: 0.9;">POMPIERS</div>
                            <div style="font-size: 14px; opacity: 0.8; margin-top: 5px;">Protection Civile</div>
                        </div>
                    </a>
                </div>
                """)
            
            with gr.Column():
                gr.HTML("""
                <div style="text-align: center; margin: 10px;">
                    <a href="tel:198" style="text-decoration: none;">
                        <div style="
                            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                            color: white;
                            padding: 20px;
                            border-radius: 12px;
                            cursor: pointer;
                            font-size: 18px;
                            box-shadow: 0 6px 20px rgba(52, 152, 219, 0.3);
                            transition: all 0.3s ease;
                            margin: 10px;
                        " onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 12px 25px rgba(52, 152, 219, 0.4)';"
                        onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 6px 20px rgba(52, 152, 219, 0.3)';">
                            <div style="font-size: 30px; margin-bottom: 10px;">🚓</div>
                            <div style="font-size: 32px; font-weight: 800; margin: 10px 0;">198</div>
                            <div style="font-size: 16px; opacity: 0.9;">POLICE</div>
                            <div style="font-size: 14px; opacity: 0.8; margin-top: 5px;">Sécurité Nationale</div>
                        </div>
                    </a>
                </div>
                """)

        # Section hôpitaux cliquable
        gr.Markdown("""
        ### 🏥 **Hôpitaux principaux**
        """)

        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div style="
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 5px solid #27ae60;
                    margin: 10px;
                ">
                    <h4 style="margin: 0 0 10px 0;">Charles Nicolle</h4>
                    <p style="margin: 0 0 15px 0; color: #666;"><small>Tunis</small></p>
                    <a href="tel:71578000" style="text-decoration: none;">
                        <div style="
                            background: linear-gradient(135deg, #27ae60 0%, #219653 100%);
                            color: white;
                            padding: 12px;
                            border-radius: 8px;
                            text-align: center;
                            font-weight: 600;
                            cursor: pointer;
                            transition: all 0.3s ease;
                        " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(39, 174, 96, 0.3)';"
                        onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none';">
                            📞 71 578 000
                        </div>
                    </a>
                </div>
                """)
            
            with gr.Column():
                gr.HTML("""
                <div style="
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 5px solid #27ae60;
                    margin: 10px;
                ">
                    <h4 style="margin: 0 0 10px 0;">La Rabta</h4>
                    <p style="margin: 0 0 15px 0; color: #666;"><small>Tunis</small></p>
                    <a href="tel:71573000" style="text-decoration: none;">
                        <div style="
                            background: linear-gradient(135deg, #27ae60 0%, #219653 100%);
                            color: white;
                            padding: 12px;
                            border-radius: 8px;
                            text-align: center;
                            font-weight: 600;
                            cursor: pointer;
                            transition: all 0.3s ease;
                        " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(39, 174, 96, 0.3)';"
                        onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none';">
                            📞 71 573 000
                        </div>
                    </a>
                </div>
                """)

        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div style="
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 5px solid #27ae60;
                    margin: 10px;
                ">
                    <h4 style="margin: 0 0 10px 0;">Habib Thameur</h4>
                    <p style="margin: 0 0 15px 0; color: #666;"><small>Tunis</small></p>
                    <a href="tel:71391000" style="text-decoration: none;">
                        <div style="
                            background: linear-gradient(135deg, #27ae60 0%, #219653 100%);
                            color: white;
                            padding: 12px;
                            border-radius: 8px;
                            text-align: center;
                            font-weight: 600;
                            cursor: pointer;
                            transition: all 0.3s ease;
                        " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(39, 174, 96, 0.3)';"
                        onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none';">
                            📞 71 391 000
                        </div>
                    </a>
                </div>
                """)
            
            with gr.Column():
                gr.HTML("""
                <div style="
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 5px solid #27ae60;
                    margin: 10px;
                ">
                    <h4 style="margin: 0 0 10px 0;">Fattouma Bourguiba</h4>
                    <p style="margin: 0 0 15px 0; color: #666;"><small>Monastir</small></p>
                    <a href="tel:73462000" style="text-decoration: none;">
                        <div style="
                            background: linear-gradient(135deg, #27ae60 0%, #219653 100%);
                            color: white;
                            padding: 12px;
                            border-radius: 8px;
                            text-align: center;
                            font-weight: 600;
                            cursor: pointer;
                            transition: all 0.3s ease;
                        " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(39, 174, 96, 0.3)';"
                        onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none';">
                            📞 73 462 000
                        </div>
                    </a>
                </div>
                """)

        # Avertissement final
        gr.Markdown("""
        <div style="
            background: #fff3cd;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #ffc107;
            margin-top: 30px;
            text-align: center;
        ">
            <div style="font-weight: 700; color: #856404; margin-bottom: 10px;">
                ⚠️ Assistant éducatif seulement
            </div>
            <div style="color: #856404;">
                Cette application ne remplace pas une consultation médicale.<br>
                Consultez toujours un médecin pour un avis médical personnel.
            </div>
        </div>
        """)

        # CSS pour responsive design
        gr.Markdown("""
        <style>
        /* Style pour mobile */
        @media (max-width: 768px) {
            .gradio-container {
                padding: 10px !important;
            }
            
            div[style*="margin: 10px;"] {
                margin: 5px !important;
            }
            
            div[style*="padding: 20px;"] {
                padding: 15px !important;
            }
            
            div[style*="font-size: 32px;"] {
                font-size: 28px !important;
            }
            
            div[style*="font-size: 30px;"] {
                font-size: 26px !important;
            }
        }

        /* Animation au clic */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(0.95); }
            100% { transform: scale(1); }
        }

        a:active > div {
            animation: pulse 0.2s ease;
        }

        /* Amélioration de l'accessibilité */
        a:focus > div {
            outline: 3px solid #3498db;
            outline-offset: 2px;
        }
        </style>
        """)
        
        # Actions
        def clear_all():
            return "", "Prêt pour votre question..."
        
        submit_btn.click(
            fn=process_medical_input,
            inputs=input_text,
            outputs=output_text
        )
        
        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[input_text, output_text]
        )
        
        input_text.submit(
            fn=process_medical_input,
            inputs=input_text,
            outputs=output_text
        )
    
    return demo

# -----------------------------
# Point d'entrée amélioré
# -----------------------------
if __name__ == "__main__":
    # Test Ollama
    print("\n🧪 Test de connexion à Ollama...")
    try:
        test = subprocess.run(
            ["ollama", "run", MODEL_NAME, "Bonjour"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if test.stdout:
            print("✅ Phi-2.7b fonctionne correctement!")
        else:
            print("⚠️ Ollama répond mais sans sortie")
    except Exception as e:
        print(f"⚠️ Test Ollama échoué: {str(e)}")
    
    # Stats dataset
    print(f"\n📈 Dataset final: {len(df)} termes médicaux")
    print(f"   - Tests labo: {len(df[df['type'] == 'lab test']) if 'type' in df.columns else 'N/A'}")
    print(f"   - Maladies: {len(df[df['type'] == 'disease']) if 'type' in df.columns else 'N/A'}")
    print(f"   - Avec valeurs normales: {len(df[pd.notna(df['normal_min'])])}")
    
    print("\n🚀 Lancement de l'interface...")
    print("🌐 **Ouvrez votre navigateur à:** http://localhost:7862")
    print("✨ **Nouveauté:** Reconnaissance améliorée des termes et abréviations")
    
    try:
        demo = create_interface()
        demo.launch(
            server_name="127.0.0.1",
            server_port=7862,
            share=False,
            show_error=True,
            inbrowser=True
        )
    except OSError as e:
        if "address already in use" in str(e):
            print(f"⚠️ Port occupé. Essai avec le port 7863...")
            demo.launch(server_port=7863)
        else:
            raise e
    except KeyboardInterrupt:
        print("\n👋 Arrêt de l'application...")
    except Exception as e:
        print(f"\n❌ Erreur: {str(e)}")