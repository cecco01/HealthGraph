# Healthcare Knowledge Graph

## Panoramica del Progetto

Questo progetto è stato sviluppato per il corso di Business & Management e utilizza un grafo di conoscenza per analizzare e visualizzare dati sanitari. Il grafo combina informazioni sugli ospedali, servizi, medici e recensioni dei pazienti, permettendo di effettuare analisi avanzate e visualizzazioni interattive.

Il progetto utilizza dati reali provenienti da Medicare.gov, combinati con un dataset di recensioni fittizio generato in base alle caratteristiche degli ospedali.

## Caratteristiche Principali

- **Analisi degli Ospedali**: Visualizza informazioni dettagliate sugli ospedali, inclusi rating, servizi offerti e medici.
- **Analisi dei Servizi**: Esplora i servizi offerti dagli ospedali e le relative recensioni.
- **Analisi delle Recensioni**: Analizza le recensioni dei pazienti, estrae parole chiave e trova recensioni simili.
- **Visualizzazione del Grafo**: Visualizza il grafo di conoscenza in modo interattivo, con la possibilità di esplorare le relazioni tra entità.
- **Confronto tra Ospedali**: Confronta gli ospedali in base a diverse metriche.

## Struttura del Progetto

Il progetto è organizzato nei seguenti file:

- `data_preparation.py`: Gestisce il download e la preparazione dei dati.
- `kg_builder.py`: Costruisce e gestisce il grafo di conoscenza.
- `ui.py`: Interfaccia utente basata su Streamlit.
- `requirements.txt`: Dipendenze del progetto.

## Struttura del Knowledge Graph

Il Knowledge Graph è strutturato come segue:

- **Nodi**:
  - Ospedali: rappresentano gli ospedali con le loro caratteristiche.
  - Servizi: rappresentano i servizi offerti dagli ospedali.
  - Medici: rappresentano i medici che lavorano negli ospedali.
  - Recensioni: rappresentano le recensioni dei pazienti.

- **Archi**:
  - Ospedale-Servizio: rappresenta la relazione "l'ospedale offre il servizio".
  - Servizio-Recensione: rappresenta la relazione "il servizio è descritto dalla recensione".
  - Medico-Ospedale: rappresenta la relazione "il medico lavora nell'ospedale".
  - Medico-Servizio: rappresenta la relazione "il medico è specializzato nel servizio".

## Funzionalità Avanzate

### Analisi delle Recensioni

L'applicazione utilizza tecniche di elaborazione del linguaggio naturale per analizzare le recensioni:

- **Estrazione di parole chiave**: Identifica le parole più importanti nelle recensioni.
- **Recensioni simili**: Trova recensioni simili in base al contenuto.

### Confronto tra Ospedali

L'applicazione permette di confrontare gli ospedali in base a diverse metriche:

- Rating complessivo
- Metriche di qualità
- Servizi offerti

### Visualizzazione Interattiva

Il grafo di conoscenza può essere visualizzato in modo interattivo:

- **Grafo completo**: Visualizza l'intero grafo di conoscenza.
- **Grafo semplificato**: Visualizza un grafo semplificato focalizzato su un ospedale specifico.

## Come provare il progetto in locale (Windows)

(Richiede Python, versione consigliata: Python 3.11 )

3) (Consigliato) Crea un ambiente virtuale
```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```
(Se salti questo passo, i pacchetti verranno installati sul tuo Python globale)

4) Installa le dipendenze
```powershell
pip install -r requirements.txt
```

5) Avvia l'app
```powershell
streamlit run ui.py

## Autori

- Leonardo Ceccarelli
- Nicolò Bacherotti
- Chiara Masiero



