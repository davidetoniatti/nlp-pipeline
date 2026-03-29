# Scalable NLP Analysis Pipeline

Il sistema acquisisce documenti, li invia a un servizio NLP e salva in Postgres le esecuzioni e i risultati finali.

## Scopo del progetto

Per ogni documento in input, il sistema:
- raccoglie il testo,
- esegue sentiment analysis multilingua,
- estrae entità nominate (NER),
- genera un riassunto (via LLM) solo per i testi con sentiment negativo,
- salva in modo persistente run e risultati nel database per garantire tracciabilità e riproducibilità.

## Architettura

Il sistema è diviso in tre componenti principali:

- **Postgres**: salva documenti, analysis run, risultati, metadati della sorgente e versioni dei modelli (model registry minimale).
- **Backend Go (Orchestratore)**: consuma messaggi da una coda mock, costruisce batch, chiama il servizio NLP, gestisce i retry con backoff esponenziale e persiste i risultati verificandone l'integrità.
- **Servizio NLP (Python/FastAPI)**: espone un'API HTTP che esegue sentiment analysis e NER localmente (Transformers) e summarization condizionale tramite Groq/LLM.

## Avvio rapido

### Requisiti

- Docker
- Docker Compose
- Una `GROQ_API_KEY` valida (opzionale, per la summarization)

### Avvio

```bash
docker compose up --build
```

### Configurazione

Il sistema è configurabile tramite variabili di ambiente nel file `compose.yml` o in un file `.env`:

- `DATABASE_URL`: URL di connessione a Postgres per il backend Go.
- `INFERENCE_URL`: Endpoint del servizio NLP (default: `http://ai_service:8080/analyze`).
- `GROQ_API_KEY`: Chiave API per la summarization via LLM.
- `FAIL_DOC_IDS`: Lista di ID documento (separati da virgola) per simulare fallimenti iniettati.
- `FAIL_TEXT_CONTAINS`: Stringa che, se contenuta nel testo, attiva un fallimento simulato.
- `REQUEST_TIMEOUT_S`: Timeout per le richieste al servizio NLP.
- `LLM_TIMEOUT_S`: Timeout specifico per la chiamata all'LLM.

## Flusso dei dati

1. Il backend Go carica i documenti mock da `test_data.json`.
2. I documenti vengono raggruppati in batch in base alla dimensione o al tempo (flush interval).
3. Il backend chiama l'endpoint `/analyze` del servizio Python.
4. Il servizio NLP esegue:
   - Sentiment analysis multilingua (XLMR-Roberta).
   - NER multilingua (WikiNeural).
   - Summarization condizionale (Llama 3 via Groq) solo per i testi negativi.
5. Il backend riceve i risultati del batch e i metadati delle versioni dei modelli usati.
6. I risultati vengono persistiti in Postgres all'interno di una singola transazione.
7. La persistenza viene verificata interrogando il database prima di confermare (ack) il batch.

## Schema del database

Il database è progettato per supportare la tracciabilità completa di ogni analisi.

Tabelle principali:

- `source_metadata`: Informazioni sull'origine del documento.
- `document`: Il documento grezzo ricevuto in ingresso.
- `model_version`: Registro dei modelli usati (nome, versione, provider, hash del prompt).
- `analysis_run`: Record di un'esecuzione di analisi, collegato al documento e ai modelli usati.
- `analysis_result`: L'output strutturato (sentiment, entità, summary) di una run completata.

Lo schema viene inizializzato automaticamente dai file SQL in `db/init`.

## Design choices

- **Batching**: Ottimizza il throughput e riduce il numero di chiamate di rete.
- **Retry con backoff e jitter**: Gestisce errori transitori e backpressure (429) in modo resiliente.
- **Separazione delle responsabilità**: Go gestisce l'orchestrazione e la persistenza; Python isola il runtime ML e le dipendenze pesanti (torch, transformers).
- **Tracciabilità dei modelli**: Ogni risultato è collegato a una `model_version` specifica, inclusi revisione del modello e hash del prompt.
- **Resilience Simulator**: Permette di testare la robustezza della pipeline iniettando fallimenti controllati tramite variabili d'ambiente.

## Struttura del repository

```text
.
├── compose.yml           # Orchestrazione dei servizi
├── test_data.json        # Dataset di test
├── backend/              # Orchestratore in Go
│   ├── cmd/server/       # Entry point dell'applicazione
│   ├── internal/pipeline/# Core logic: batching, store, orchestrator
│   ├── Dockerfile
│   └── go.mod            # Dipendenze Go (1.25)
├── nlp_service/          # Servizio NLP in Python
│   ├── app.py            # API FastAPI
│   ├── nlp_inference.py  # Logica di inferenza e modelli
│   └── requirements.txt  # Dipendenze Python (versioni pinned)
└── db/init/              # Script di inizializzazione SQL
```

## API

Il servizio NLP espone:
- `POST /analyze`: Accetta batch di documenti e restituisce analisi e metadati dei modelli.
- `GET /health`: Verifica la readiness del servizio e il caricamento dei modelli.
