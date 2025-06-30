---
marp: true
theme: default
paginate: true
header: "Healthcare Knowledge Graph"
footer: "Presentazione Progetto"
---

# Healthcare Knowledge Graph
## Analisi dei Servizi Ospedalieri attraverso Grafi di Conoscenza

---

# Indice
1. Introduzione
2. Architettura del Sistema
3. Implementazione
4. Tecnologie Utilizzate
5. Funzionalità
6. Risultati
7. Conclusioni

---

# Introduzione

## Scopo del Progetto
- Applicazione web interattiva per l'analisi dei dati ospedalieri
- Visualizzazione delle relazioni tra ospedali e servizi
- Strumento per la comprensione dei dati sanitari

## Obiettivi Principali
- Creazione di un grafo di conoscenza
- Sviluppo di un'interfaccia utente intuitiva
- Implementazione di funzionalità di analisi
- Strumenti per il confronto tra ospedali

---

# Architettura del Sistema

## Componenti Principali
- **Data Preparation**: Pulizia e standardizzazione dati
- **Knowledge Graph Builder**: Costruzione del grafo
- **User Interface**: Visualizzazione e interazione

## Flusso dei Dati
1. Lettura dati CSV
2. Pulizia e standardizzazione
3. Costruzione grafo
4. Visualizzazione interattiva

---

# Implementazione

## Preparazione Dati
- Lettura dati ospedali
- Gestione valori mancanti
- Standardizzazione nomi colonne
- Salvataggio dati puliti

## Grafo di Conoscenza
- Nodi: ospedali, servizi, stati, tipi
- Relazioni tra nodi
- Assegnazione servizi
- Analisi e esportazione

---

# Tecnologie Utilizzate

## Framework e Librerie
- **Streamlit**: Interfaccia web
- **NetworkX**: Gestione grafi
- **Pandas**: Analisi dati
- **Plotly**: Visualizzazione
- **Scikit-learn**: Clustering

## Struttura Progetto
```
HealthcareKnowledgeGraph/
├── data/
├── src/
├── requirements.txt
└── README.md
```

---

# Funzionalità

## Interfaccia Utente
- Visualizzazione interattiva
- Filtrazione per tipo e stato
- Opzioni di layout
- Statistiche e metriche

## Analisi Dati
- Panoramica ospedali
- Distribuzione servizi
- Confronto strutture
- Metriche di densità

---

# Risultati

## Visualizzazione Grafo
- Identificazione servizi
- Analisi distribuzione geografica
- Confronto tipi di strutture
- Valutazione densità servizi

## Metriche Principali
- Numero ospedali per stato
- Distribuzione tipi
- Servizi per ospedale
- Densità del grafo

---

# Miglioramenti Futuri

## Potenziali Sviluppi
- Integrazione dati aggiuntivi
- Algoritmi di raccomandazione
- Analisi predittive
- Ottimizzazione performance

## Considerazioni
- Scalabilità sistema
- Performance grafo
- Caching dati
- Interfaccia utente

---

# Conclusioni

## Benefici
- Visualizzazione intuitiva
- Analisi dati efficiente
- Strumento decisionale
- Flessibilità di utilizzo

## Prospettive
- Applicazioni future
- Possibili integrazioni
- Sviluppi potenziali
- Impatto sul settore

---

# Grazie per l'attenzione
## Domande? 