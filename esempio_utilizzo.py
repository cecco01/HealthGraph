#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Esempio di utilizzo del Knowledge Graph Builder per l'analisi di dati sanitari.
Questo script mostra come utilizzare le principali funzionalità del Knowledge Graph Builder.
"""

import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from kg_builder import KnowledgeGraph
from data_preparation import download_hospital_data, prepare_hospital_data, create_reviews_dataset
import json

def main():
    """
    Funzione principale che mostra come utilizzare il Knowledge Graph Builder.
    """
    print("=" * 80)
    print("ESEMPIO DI UTILIZZO DEL KNOWLEDGE GRAPH BUILDER")
    print("=" * 80)
    
    # Verifica se i file esistono, altrimenti li scarica e li prepara
    if not os.path.exists("data/reviews.csv") or not os.path.exists("data/hospital_data_clean.csv"):
        print("Download e preparazione dei dati in corso...")
        
        # Crea la directory data se non esiste
        if not os.path.exists("data"):
            os.makedirs("data")
        
        # Scarica e prepara i dati
        if download_hospital_data():
            df = prepare_hospital_data()
            if df is not None:
                create_reviews_dataset(df)
                print("Dati scaricati e preparati con successo!")
            else:
                print("Errore nella preparazione dei dati.")
                return
        else:
            print("Errore nel download dei dati.")
            return
    
    # Inizializza il Knowledge Graph
    print("\nInizializzazione del Knowledge Graph...")
    kg = KnowledgeGraph(
        reviews_path="data/reviews.csv",
        hospital_data_path="data/hospital_data_clean.csv"
    )
    
    # Esempio di analisi del testo
    print("\n1. ANALISI DEL TESTO DI UNA RECENSIONE")
    print("-" * 50)
    sample_review = """
    Ho avuto un'esperienza eccellente all'Ospedale San Giovanni. 
    Il Dottor Rossi è stato molto gentile e professionale.
    La pulizia delle camere era impeccabile. 
    L'unico punto negativo è stato il tempo di attesa in pronto soccorso.
    """
    text_analysis = kg.analyze_text(sample_review)
    print(json.dumps(text_analysis, indent=2, ensure_ascii=False))
    
    # Esempio di analisi del sentiment
    print("\n2. ANALISI DEL SENTIMENT")
    print("-" * 50)
    sentiment = kg.get_sentiment_analysis(sample_review)
    print(json.dumps(sentiment, indent=2, ensure_ascii=False))
    
    # Esempio di estrazione di parole chiave
    print("\n3. ESTRAZIONE DI PAROLE CHIAVE")
    print("-" * 50)
    keywords = kg.get_text_keywords(sample_review, top_k=5)
    print(f"Parole chiave: {', '.join(keywords)}")
    
    # Esempio di preprocessing del testo
    print("\n4. PREPROCESSING DEL TESTO")
    print("-" * 50)
    processed_text = kg.preprocess_review(sample_review)
    print(f"Testo originale: {sample_review}")
    print(f"Testo preprocessato: {processed_text}")
    
    # Ottieni informazioni sul grafo
    print(f"\nGrafo costruito con {kg.graph.number_of_nodes()} nodi e {kg.graph.number_of_edges()} archi")
    
    # Ottieni tutti gli ospedali
    hospitals = kg.get_hospitals()
    print(f"\nNumero di ospedali: {len(hospitals)}")
    
    # Ottieni i migliori ospedali
    print("\nMigliori ospedali (rating più alto):")
    top_hospitals = kg.get_top_hospitals(top_k=5)
    for hospital, rating in top_hospitals:
        print(f"- {hospital}: {rating:.2f}/5")
    
    # Ottieni gli ospedali in uno stato specifico
    states = kg.get_states()
    if states:
        state = states[0]  # Prendi il primo stato disponibile
        print(f"\nOspedali nello stato {state}:")
        hospitals_in_state = kg.get_hospitals_by_state(state)
        for hospital in hospitals_in_state[:5]:  # Mostra solo i primi 5
            print(f"- {hospital}")
    
    # Ottieni i tipi di ospedale
    hospital_types = kg.get_hospital_types()
    print(f"\nTipi di ospedale disponibili: {', '.join(hospital_types)}")
    
    # Ottieni i servizi di un ospedale
    if hospitals:
        hospital = hospitals[0]  # Prendi il primo ospedale disponibile
        print(f"\nServizi offerti da {hospital}:")
        services = kg.get_services(hospital)
        for service in services:
            print(f"- {service}")
    
    # Ottieni i medici di un ospedale
    if hospitals:
        print(f"\nMedici che lavorano in {hospital}:")
        doctors = kg.get_doctors(hospital)
        for doctor in doctors:
            print(f"- {doctor}")
    
    # Ottieni le recensioni di un servizio
    if services:
        service = services[0]  # Prendi il primo servizio disponibile
        print(f"\nRecensioni per il servizio {service}:")
        reviews = kg.get_reviews(service)
        for review in reviews[:3]:  # Mostra solo le prime 3 recensioni
            print(f"- {review}")
    
    # Ottieni il rating di un ospedale
    if hospitals:
        print(f"\nRating di {hospital}: {kg.get_hospital_rating(hospital):.2f}/5")
    
    # Ottieni il rating di un servizio
    if services:
        print(f"\nRating del servizio {service}: {kg.get_service_rating(service):.2f}/5")
    
    # Ottieni il rating di un medico
    if doctors:
        doctor = doctors[0]  # Prendi il primo medico disponibile
        print(f"\nRating del medico {doctor}: {kg.get_doctor_rating(doctor):.2f}/5")
    
    # Trova recensioni simili
    if reviews:
        review = reviews[0]  # Prendi la prima recensione disponibile
        print(f"\nRecensioni simili a: '{review}'")
        similar_reviews = kg.get_similar_reviews(review, top_k=3)
        for similar_review in similar_reviews:
            print(f"- {similar_review}")
    
    # Estrai parole chiave da una recensione
    if reviews:
        print(f"\nParole chiave estratte dalla recensione: '{review}'")
        keywords = kg.extract_keywords(review, top_k=5)
        print(f"- {', '.join(keywords)}")
    
    # Confronta due ospedali
    if len(hospitals) >= 2:
        hospital1, hospital2 = hospitals[0], hospitals[1]
        print(f"\nConfronto tra {hospital1} e {hospital2}:")
        comparison = kg.compare_hospitals([hospital1, hospital2])
        
        if comparison:
            # Mostra solo le prime 5 metriche per brevità
            for i, (metric, values) in enumerate(comparison.items()):
                if i >= 5:
                    break
                print(f"- {metric}:")
                for hospital, value in values.items():
                    print(f"  - {hospital}: {value}")
    
    # Esporta il grafo in formato JSON
    print("\nEsportazione del grafo in formato JSON...")
    kg.export_graph_to_json("data/graph_visualization.json")
    print("Grafo esportato con successo!")
    
    print("\n" + "=" * 80)
    print("ESEMPIO COMPLETATO")
    print("=" * 80)

if __name__ == "__main__":
    main() 