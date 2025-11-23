import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import Credentials

app = Flask(__name__)
CORS(app)

# ==========================================
# 🔑 VOS IDENTIFIANTS IBM (A REMPLIR ICI)
# ==========================================
API_KEY = os.getenv("IBM_API_KEY") 
PROJECT_ID = os.getenv("IBM_PROJECT_ID")
REGION_URL = "https://us-south.ml.cloud.ibm.com"  # Dallas (laissez tel quel vu vos captures)

# ==========================================
# 🧠 CONFIGURATION DU CERVEAU (IBM GRANITE)
# ==========================================
# On utilise IBM Granite pour maximiser les points "Utilisation Tech IBM" auprès des juges.
# Si besoin, vous pouvez changer pour "meta-llama/llama-3-70b-instruct"
MODEL_ID = "ibm/granite-3-3-8b-instruct"

def get_ibm_analysis(text_contract):
    print("🧠 Envoi du contrat au modèle IBM watsonx...")
    
    # 1. Authentification
    creds = Credentials(url=REGION_URL, api_key=API_KEY)
    
    # 2. Paramètres de génération (Créativité basse pour la rigueur juridique)
    params = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MAX_NEW_TOKENS: 900,
        GenParams.MIN_NEW_TOKENS: 10,
        GenParams.REPETITION_PENALTY: 1.1
    }

    # 3. Initialisation du modèle
    model = ModelInference(
        model_id=MODEL_ID,
        params=params,
        credentials=creds,
        project_id=PROJECT_ID
    )

    # 4. Le Prompt "Avocat Senior" (Optimisé pour sortir du JSON strict)
    prompt = f"""
    You are a senior M&A Lawyer AI. Analyze the following contract text strictly.
    
    Your goal is to identify risks and output the result in a VALID JSON format only. Do not output markdown.
    
    Input Text:
    "{text_contract}"
    
    Instructions:
    1. Identify a risk score (0 to 100).
    2. List critical flags (clauses like Penalty, Termination, Exclusive, Liability).
    3. Write a short executive summary.
    
    Output Format (JSON ONLY):
    {{
        "risk_score": <integer>,
        "critical_flags": ["flag1", "flag2"],
        "executive_summary": "<text>"
    }}
    
    JSON Response:
    """

    # 5. Génération
    response_text = model.generate_text(prompt=prompt)
    print(f"🤖 Réponse brute IBM: {response_text}")
    
    # Nettoyage basique pour être sûr d'avoir du JSON propre
    import json
    import re
    try:
        # On essaie de trouver le JSON dans la réponse (au cas où l'IA bavarde un peu avant)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return json.loads(response_text)
    except:
        # Fallback de sécurité si l'IA échoue à faire du JSON parfait
        return {
            "risk_score": 50,
            "critical_flags": ["Erreur Format IA"],
            "executive_summary": "L'IA a analysé le texte mais le format de sortie était invalide. Texte brut: " + response_text[:100]
        }

@app.route('/analyze_risk', methods=['POST'])
def analyze_risk():
    try:
        data = request.get_json()
        contract_text = data.get('contract_text', '')[:3000] # On coupe à 3000 caractères pour la démo (limite de tokens)
        
        if not contract_text:
            return jsonify({"error": "No text provided"}), 400

        # Appel à la vraie IA
        analysis_result = get_ibm_analysis(contract_text)
        
        return jsonify(analysis_result), 200

    except Exception as e:
        print(f"🔥 Erreur Critique: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Serveur DealFlow AI (Powered by IBM Granite) en écoute...")
    app.run(host='0.0.0.0', port=5000, debug=True)