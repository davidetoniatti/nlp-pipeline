# Case study: Scalable NLP Analysis Pipeline

## Problema
La nostra azienda deve processare un flusso continuo di feedback testuali (commenti, recensioni, chat) provenienti da diverse sorgenti. Dobbiamo classsificare il sentiment, estrarre entità nominate (NER) e generare un riassunto automatico utilizzando modelli di GenAI.

## Obiettivo della prova
Progettare e implementare un prototipo (o una architettura dettagliata) che dimostri come gestire questo carico di lavoro in modo scalabile e resiliente.

## Task 1: Architettura e Data Design
1. Diagramma relazionale: disegna (testualmente o con tool tipo Lucidchart/Mermaid) uno schema di db SQL per memorizzare:
    - I metadati della sorgente
    - Il testo originale
    - I risultati della analisi (sentiment score, entità, summary)
    - Versionamento modello AI utilizzato (fondamentale per la tracciabilità)
2. Data Structure: Definisci la struttura dati ottimale in Golang per rappresentare un "Batch" di documenti che deve essere processato per minimizzare l'allocazione di memoria.

## Task 2: Sviluppo AI (python e pytorch)

Fornisci uno script Python che:
1. carichi un modello Transformer (DistillBERT o un modello da HuggingFace) usando pytorch
2. implementi una funzione di inferenza che gestisca il batching dinamico per ottimizzare l'uso della GPU/CPU
3. NLP e GenAI: scrivi un prompt o una logica di "Chain of Thought" per un LLM che riassuma il testo slo se il sentiment rilevato è "Negativo"

## Task 3: Concurrency e Backend (Golang)
Implementa (o descrivi con un codice Go) un Orchestrator che:
1. Legga i dati da una coda (es. un canale o un mock di RabbitMQ/Kafka)
2. Distribuisca il carico a N "Worker" concorrenti
3. Interfacci i Worker con lo script Python (tramite gRPC o chiamate http async)
4. gestisca i timeout e i retry in caso di fallimento del modello AI

## Task 4: Analisi predittiva (Data science)
Immagina di avere 6 mesi di dati storici sui feedback. Spiega come proggetteresti un modello predittivo capace di avvisare il team di supporto se si prevede un picco di feedback negativi nelle prossime 24 ore. Quali feature useresti? Quale algoritmo sceglieresti?
