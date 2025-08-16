import openai
import requests
from typing import Dict, List, Tuple
import networkx as nx
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Carica il modello italiano di spaCy
# nlp = spacy.load("it_core_news_lg") # This line is removed as per the edit hint

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
    # Aggiungi nodi e archi secondo la logica del knowledge graph sanitario
    # ...
    return G 