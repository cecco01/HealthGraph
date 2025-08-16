"""
Healthcare Knowledge Graph - Preparazione Dati
===========================================

Questo file contiene le funzioni per la preparazione e pulizia dei dati degli ospedali
prima della costruzione del grafo di conoscenza.

Funzionalità principali:
- Lettura dei dati degli ospedali
- Pulizia e standardizzazione dei dati
- Gestione dei valori mancanti
- Salvataggio dei dati puliti
"""

import pandas as pd

def prepare_data():
    """
    Prepara i dataset per l'analisi utilizzando i file esistenti.
    """
    print("Preparazione dei dataset in corso...")
    
    try:
        # Leggi il dataset degli ospedali con diverse codifiche
        print("Lettura del dataset ospedali...")
        try:
            # Prova prima con latin-1
            ospedali_df = pd.read_csv("data/Hospital General Information.csv", encoding='latin-1')
        except:
            try:
                # Se non funziona, prova con cp1252
                ospedali_df = pd.read_csv("data/Hospital General Information.csv", encoding='cp1252')
            except:
                # Se ancora non funziona, prova con iso-8859-1
                ospedali_df = pd.read_csv("data/Hospital General Information.csv", encoding='iso-8859-1')
        
        # Pulisci i dati degli ospedali
        ospedali_df = ospedali_df.rename(columns={
            'Hospital Name': 'name',
            'Hospital Type': 'type',
            'Address': 'address',
            'Phone Number': 'phone',
            'Hospital overall rating': 'rating'
        })
        
        # Seleziona e rinomina le colonne necessarie
        ospedali_df = ospedali_df[['name', 'type', 'address', 'phone', 'rating']]
        
        # Aggiungi l'ID ospedale
        ospedali_df['hospital_id'] = [f'HOSP_{i+1}' for i in range(len(ospedali_df))]
        
        # Leggi il dataset delle recensioni
        print("Lettura del dataset recensioni...")
        try:
            # Prova prima con latin-1
            recensioni_df = pd.read_csv("data/reviews.csv", encoding='latin-1')
        except:
            try:
                # Se non funziona, prova con cp1252
                recensioni_df = pd.read_csv("data/reviews.csv", encoding='cp1252')
            except:
                # Se ancora non funziona, prova con iso-8859-1
                recensioni_df = pd.read_csv("data/reviews.csv", encoding='iso-8859-1')
        
        # Salva i dataset puliti
        ospedali_df.to_csv('data/hospital_data_clean.csv', index=False, encoding='utf-8')
        recensioni_df.to_csv('data/reviews_clean.csv', index=False, encoding='utf-8')
        
        print(f"Dataset preparati con successo:")
        print(f"- {len(ospedali_df)} ospedali")
        print(f"- {len(recensioni_df)} recensioni")
        print("I file sono stati salvati nella cartella 'data'")
        
    except Exception as e:
        print(f"Errore durante la preparazione dei dataset: {str(e)}")
        print("Assicurati che i file 'Hospital General Information.csv' e 'reviews.csv' siano presenti nella cartella 'data'")

if __name__ == "__main__":
    prepare_data() 