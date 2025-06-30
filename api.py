from flask import Flask, jsonify, request
import networkx as nx
from kg_builder import KnowledgeGraph
import openai
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()

# Configurazione API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
kg = KnowledgeGraph("data/Hospital General Information.csv")

@app.route('/get_hospitals', methods=['GET'])
def get_hospitals():
    hospitals = [n for n, attr in kg.get_graph().nodes(data=True) if attr["type"] == "Hospital"]
    return jsonify(hospitals)

@app.route('/get_services/<hospital>', methods=['GET'])
def get_services(hospital):
    services = [n for n in kg.get_graph().neighbors(hospital) if kg.get_graph().nodes[n]["type"] == "Service"]
    return jsonify(services)

@app.route('/get_reviews/<service>', methods=['GET'])
def get_reviews(service):
    reviews = [n for n in kg.get_graph().neighbors(service) if kg.get_graph().nodes[n]["type"] == "Review"]
    return jsonify(reviews)

@app.route('/get_doctors/<hospital>', methods=['GET'])
def get_doctors(hospital):
    doctors = [n for n in kg.get_graph().neighbors(hospital) if kg.get_graph().nodes[n]["type"] == "Doctor"]
    return jsonify(doctors)

@app.route('/analyze_reviews', methods=['POST'])
def analyze_reviews():
    data = request.json
    hospital = data.get('hospital')
    service = data.get('service')
    ai_model = data.get('ai_model', 'openai')  # Default a OpenAI
    
    # Ottieni tutte le recensioni per l'ospedale e il servizio
    reviews = []
    if hospital and service:
        # Trova il servizio specifico dell'ospedale
        hospital_services = [n for n in kg.get_graph().neighbors(hospital) if kg.get_graph().nodes[n]["type"] == "Service"]
        if service in hospital_services:
            reviews = [n for n in kg.get_graph().neighbors(service) if kg.get_graph().nodes[n]["type"] == "Review"]
    
    if not reviews:
        return jsonify({"error": "Nessuna recensione trovata"}), 404
    
    # Analizza le recensioni con l'AI selezionata
    if ai_model == 'openai':
        return analyze_with_openai(reviews)
    elif ai_model == 'gemini':
        return analyze_with_gemini(reviews)
    else:
        return jsonify({"error": "Modello AI non supportato"}), 400

def analyze_with_openai(reviews):
    prompt = f"""Analizza le seguenti recensioni di un ospedale e fornisci:
    1. Un riassunto generale del sentiment (positivo/negativo/neutro)
    2. I punti di forza menzionati
    3. Le aree di miglioramento suggerite
    4. Raccomandazioni per il personale sanitario
    
    Recensioni:
    {', '.join(reviews)}
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Sei un esperto di analisi delle recensioni sanitarie."},
                {"role": "user", "content": prompt}
            ]
        )
        
        analysis = response["choices"][0]["message"]["content"]
        return jsonify({"analysis": analysis, "model": "OpenAI GPT-4"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def analyze_with_gemini(reviews):
    prompt = f"""Analizza le seguenti recensioni di un ospedale e fornisci:
    1. Un riassunto generale del sentiment (positivo/negativo/neutro)
    2. I punti di forza menzionati
    3. Le aree di miglioramento suggerite
    4. Raccomandazioni per il personale sanitario
    
    Recensioni:
    {', '.join(reviews)}
    """
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return jsonify({"analysis": response.text, "model": "Google Gemini"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_graph_data', methods=['GET'])
def get_graph_data():
    """Restituisce i dati del grafo in formato JSON per la visualizzazione"""
    graph = kg.get_graph()
    
    nodes = []
    edges = []
    
    for node, attr in graph.nodes(data=True):
        nodes.append({
            "id": node,
            "type": attr.get("type", "Unknown"),
            "label": node
        })
    
    for source, target, attr in graph.edges(data=True):
        edges.append({
            "source": source,
            "target": target,
            "relation": attr.get("relation", "connected")
        })
    
    return jsonify({
        "nodes": nodes,
        "edges": edges
    })

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    """
    Endpoint per l'analisi del testo di una recensione
    """
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({'error': 'Testo mancante'}), 400
    
    try:
        analysis = kg.analyze_text(text)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_sentiment', methods=['POST'])
def get_sentiment():
    """
    Endpoint per l'analisi del sentiment di un testo
    """
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({'error': 'Testo mancante'}), 400
    
    try:
        sentiment = kg.get_sentiment_analysis(text)
        return jsonify(sentiment)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_keywords', methods=['POST'])
def get_keywords():
    """
    Endpoint per l'estrazione delle parole chiave da un testo
    """
    data = request.json
    text = data.get('text')
    top_k = data.get('top_k', 5)
    
    if not text:
        return jsonify({'error': 'Testo mancante'}), 400
    
    try:
        keywords = kg.get_text_keywords(text, top_k)
        return jsonify({'keywords': keywords})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/preprocess_text', methods=['POST'])
def preprocess_text():
    """
    Endpoint per il preprocessing di un testo
    """
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({'error': 'Testo mancante'}), 400
    
    try:
        processed_text = kg.preprocess_review(text)
        return jsonify({'processed_text': processed_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

