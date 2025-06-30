import openai
import json
import requests
from typing import Dict, List, Tuple, Set
import spacy
import networkx as nx
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Carica il modello italiano di spaCy
nlp = spacy.load("it_core_news_lg")

def preprocess_text(text: str) -> str:
    """
    Preprocessa il testo rimuovendo caratteri speciali e normalizzando.
    
    Parameters:
    text (str): Testo da preprocessare
    
    Returns:
    str: Testo preprocessato
    """
    # Rimuovi caratteri speciali e converti in minuscolo
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Rimuovi spazi multipli
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Estrae entità dal testo usando spaCy.
    
    Parameters:
    text (str): Testo da analizzare
    
    Returns:
    Dict[str, List[str]]: Dizionario con entità estratte per categoria
    """
    doc = nlp(text)
    entities = {
        "PERSON": [],
        "ORG": [],
        "GPE": [],  # Geopolitical entities (luoghi)
        "PRODUCT": [],
        "FAC": []   # Facilities (strutture)
    }
    
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    
    return entities

def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analizza il sentiment del testo usando regole e dizionari.
    
    Parameters:
    text (str): Testo da analizzare
    
    Returns:
    Dict[str, float]: Dizionario con punteggi di sentiment
    """
    # Dizionari di parole positive e negative in italiano
    positive_words = set(['eccellente', 'ottimo', 'buono', 'gentile', 'professionale', 
                         'pulito', 'efficiente', 'competente', 'cortese', 'attento'])
    negative_words = set(['pessimo', 'terribile', 'scadente', 'disorganizzato', 
                         'trascurato', 'lento', 'incompetente', 'maleducato'])
    
    words = text.lower().split()
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)
    total = len(words)
    
    return {
        "positive": pos_count / total if total > 0 else 0,
        "negative": neg_count / total if total > 0 else 0,
        "neutral": (total - pos_count - neg_count) / total if total > 0 else 0
    }

def extract_keywords(text: str, top_k: int = 5) -> List[str]:
    """
    Estrae le parole chiave dal testo usando TF-IDF.
    
    Parameters:
    text (str): Testo da analizzare
    top_k (int): Numero di parole chiave da estrarre
    
    Returns:
    List[str]: Lista delle parole chiave
    """
    vectorizer = TfidfVectorizer(max_features=100, stop_words='italian')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    
    # Ottieni i pesi TF-IDF
    weights = tfidf_matrix.toarray()[0]
    
    # Ordina le parole per peso
    sorted_indices = weights.argsort()[::-1]
    return [feature_names[i] for i in sorted_indices[:top_k]]

def find_similar_reviews(review: str, all_reviews: List[str], top_k: int = 3) -> List[str]:
    """
    Trova recensioni simili usando cosine similarity.
    
    Parameters:
    review (str): Recensione di riferimento
    all_reviews (List[str]): Lista di tutte le recensioni
    top_k (int): Numero di recensioni simili da trovare
    
    Returns:
    List[str]: Lista delle recensioni più simili
    """
    vectorizer = TfidfVectorizer(stop_words='italian')
    tfidf_matrix = vectorizer.fit_transform([review] + all_reviews)
    
    # Calcola la similarità coseno
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    
    # Trova gli indici delle recensioni più simili
    similar_indices = similarities.argsort()[::-1][:top_k]
    
    return [all_reviews[i] for i in similar_indices]

def build_knowledge_graph(reviews: List[str]) -> nx.Graph:
    """
    Costruisce un knowledge graph dalle recensioni usando networkx.
    
    Parameters:
    reviews (List[str]): Lista di recensioni
    
    Returns:
    nx.Graph: Grafo della conoscenza
    """
    G = nx.Graph()
    
    for review in reviews:
        # Estrai entità
        entities = extract_entities(review)
        
        # Aggiungi nodi per ogni entità
        for category, items in entities.items():
            for item in items:
                G.add_node(item, type=category)
        
        # Aggiungi nodo per la recensione
        review_id = f"review_{hash(review)}"
        G.add_node(review_id, type="REVIEW")
        
        # Aggiungi archi tra recensione e entità
        for category, items in entities.items():
            for item in items:
                G.add_edge(review_id, item, relation=f"MENTIONS_{category}")
        
        # Analizza sentiment
        sentiment = analyze_sentiment(review)
        G.nodes[review_id]["sentiment"] = sentiment
        
        # Estrai keywords
        keywords = extract_keywords(review)
        G.nodes[review_id]["keywords"] = keywords
        
        # Aggiungi archi tra keywords e recensione
        for keyword in keywords:
            G.add_edge(review_id, keyword, relation="CONTAINS_KEYWORD")
    
    return G

def analyze_hospital_review(review_text: str) -> Dict:
    """
    Analizza una recensione ospedaliera combinando diversi metodi.
    
    Parameters:
    review_text (str): Il testo della recensione dell'ospedale
    
    Returns:
    Dict: Dizionario contenente l'analisi completa
    """
    # Preprocessa il testo
    processed_text = preprocess_text(review_text)
    
    # Estrai entità
    entities = extract_entities(processed_text)
    
    # Analizza sentiment
    sentiment = analyze_sentiment(processed_text)
    
    # Estrai keywords
    keywords = extract_keywords(processed_text)
    
    return {
        "entities": entities,
        "sentiment": sentiment,
        "keywords": keywords,
        "processed_text": processed_text
    }

# Test della funzionalità
if __name__ == "__main__":
    test_review = """
    Ho avuto un'esperienza eccellente all'Ospedale San Giovanni. 
    Il Dottor Rossi è stato molto gentile e professionale.
    La pulizia delle camere era impeccabile. 
    L'unico punto negativo è stato il tempo di attesa in pronto soccorso.
    """
    
    # Test dell'analisi completa
    result = analyze_hospital_review(test_review)
    print("\n📊 Analisi Recensione Ospedale:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Test del knowledge graph
    reviews = [test_review]
    G = build_knowledge_graph(reviews)
    print("\n🔍 Knowledge Graph:")
    print(f"Numero di nodi: {G.number_of_nodes()}")
    print(f"Numero di archi: {G.number_of_edges()}")
    print("\nNodi nel grafo:")
    for node in G.nodes(data=True):
        print(f"- {node[0]}: {node[1]}") 