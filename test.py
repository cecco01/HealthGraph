#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test delle funzionalità del Knowledge Graph per l'analisi delle recensioni ospedaliere.
Questo script mostra come utilizzare le principali funzionalità implementate in ai_logic.py.
"""

import json
import matplotlib.pyplot as plt
import networkx as nx
from ai_logic import (
    analyze_hospital_review,
    build_knowledge_graph,
    find_similar_reviews,
    extract_keywords,
    analyze_sentiment
)

def main():
    """
    Funzione principale che mostra come utilizzare le funzionalità del Knowledge Graph.
    """
    print("=" * 80)
    print("TEST DELLE FUNZIONALITÀ DEL KNOWLEDGE GRAPH")
    print("=" * 80)
    
    # Esempio di recensioni ospedaliere
    reviews = [
        """
        Ho avuto un'esperienza eccellente all'Ospedale San Giovanni. 
        Il Dottor Rossi è stato molto gentile e professionale.
        La pulizia delle camere era impeccabile. 
        L'unico punto negativo è stato il tempo di attesa in pronto soccorso.
        """,
        
        """
        L'Ospedale Santa Maria è molto organizzato. 
        Gli infermieri sono attenti e premurosi.
        Ho apprezzato molto la rapidità delle visite mediche.
        Il cibo lascia molto a desiderare.
        """,
        
        """
        Al Policlinico Umberto I ho trovato personale poco competente.
        I tempi di attesa sono inaccettabili.
        Le strutture sono vecchie e necessitano di manutenzione.
        Non consiglio questo ospedale a nessuno.
        """
    ]
    
    # Analisi di una singola recensione
    print("\n1. ANALISI DI UNA SINGOLA RECENSIONE")
    print("-" * 50)
    result = analyze_hospital_review(reviews[0])
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Analisi del sentiment di tutte le recensioni
    print("\n2. ANALISI DEL SENTIMENT DI TUTTE LE RECENSIONI")
    print("-" * 50)
    for i, review in enumerate(reviews):
        sentiment = analyze_sentiment(review)
        print(f"Recensione {i+1}:")
        print(f"  Positivo: {sentiment['positive']:.2f}")
        print(f"  Negativo: {sentiment['negative']:.2f}")
        print(f"  Neutro: {sentiment['neutral']:.2f}")
    
    # Estrazione di parole chiave
    print("\n3. ESTRAZIONE DI PAROLE CHIAVE")
    print("-" * 50)
    for i, review in enumerate(reviews):
        keywords = extract_keywords(review, top_k=5)
        print(f"Recensione {i+1}: {', '.join(keywords)}")
    
    # Ricerca di recensioni simili
    print("\n4. RICERCA DI RECENSIONI SIMILI")
    print("-" * 50)
    similar_reviews = find_similar_reviews(reviews[0], reviews[1:], top_k=2)
    print(f"Recensioni simili alla prima:")
    for i, review in enumerate(similar_reviews):
        print(f"  {i+1}. {review[:100]}...")
    
    # Costruzione del knowledge graph
    print("\n5. COSTRUZIONE DEL KNOWLEDGE GRAPH")
    print("-" * 50)
    G = build_knowledge_graph(reviews)
    print(f"Numero di nodi: {G.number_of_nodes()}")
    print(f"Numero di archi: {G.number_of_edges()}")
    
    # Visualizzazione del grafo
    print("\n6. VISUALIZZAZIONE DEL GRAFO")
    print("-" * 50)
    print("Creazione del grafo in corso...")
    
    # Crea un grafo più piccolo per la visualizzazione
    subgraph_nodes = list(G.nodes())[:20]  # Limita a 20 nodi per la visualizzazione
    subgraph = G.subgraph(subgraph_nodes)
    
    # Disegna il grafo
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(subgraph, seed=42)
    
    # Disegna i nodi
    nx.draw_networkx_nodes(subgraph, pos, node_size=700, node_color='lightblue')
    
    # Disegna gli archi
    nx.draw_networkx_edges(subgraph, pos, alpha=0.5)
    
    # Aggiungi le etichette
    nx.draw_networkx_labels(subgraph, pos, font_size=10)
    
    plt.title("Knowledge Graph delle Recensioni Ospedaliere (sottografo)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("knowledge_graph.png")
    print("Grafo salvato come 'knowledge_graph.png'")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETATO")
    print("=" * 80)

if __name__ == "__main__":
    main() 