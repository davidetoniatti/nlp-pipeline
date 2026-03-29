# Scalable NLP Analysis Pipeline


> [!IMPORTANT]
> Lo svolgimento del task 4 si trova all'interno del file `analisi_predittiva.md`.

Per ogni documento in input, il sistema:
- raccoglie il testo (multilingua: italiano e inglese),
- esegue sentiment analysis,
- estrae entità nominate (NER),
- genera un riassunto (via LLM) solo per i testi con sentiment negativo,
- salva in modo persistente run e risultati nel database per garantire tracciabilità e riproducibilità.

## Architettura

- **Postgres**: salva documenti, analysis run, risultati, metadati della sorgente e versioni dei modelli (model registry minimale).
- **Backend Go (Orchestratore)**: consuma messaggi da una coda mock, costruisce batch, chiama il servizio NLP, gestisce i retry con backoff esponenziale e persiste i risultati verificandone l'integrità.
- **Servizio NLP (Python/FastAPI)**: espone un'API HTTP che esegue sentiment analysis e NER localmente (Transformers) e summarization condizionale tramite Groq/LLM.

## Deploy e avvio

### Requisiti

- Docker
- Docker Compose
- Una `GROQ_API_KEY` valida (opzionale, per la summarization)

### Avvio

```bash
docker compose up --build
```

### Configurazione

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

Tabelle:

- `source_metadata`: Informazioni sull'origine del documento.
- `document`: Il documento grezzo ricevuto in ingresso.
- `model_version`: Registro dei modelli usati (nome, versione, provider, hash del prompt).
- `analysis_run`: Record di un'esecuzione di analisi, collegato al documento e ai modelli usati.
- `analysis_result`: L'output strutturato (sentiment, entità, summary) di una run completata.


## Design choices

- **Batching**: Ottimizza il throughput e riduce il numero di chiamate di rete.
- **Retry con backoff e jitter**: Gestisce errori transitori e backpressure (429) in modo resiliente.
- **Tracciabilità dei modelli**: Ogni risultato è collegato a una `model_version` specifica, inclusi revisione del modello e hash del prompt.

## Struttura del repository

```text
.
├── compose.yml
├── test_data.json
├── backend/
│   ├── cmd/server/       # Entry point dell'applicazione
│   ├── internal/pipeline/# batching, store, orchestrator
│   ├── Dockerfile
│   └── go.mod
├── nlp_service/          # NLP pipeline
│   ├── app.py
│   ├── nlp_inference.py
│   └── requirements.txt
└── db/init/
```

## API

Il servizio NLP espone:
- `POST /analyze`: Accetta batch di documenti e restituisce analisi e metadati dei modelli.
- `GET /health`: Verifica la readiness del servizio e il caricamento dei modelli.

