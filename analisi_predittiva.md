# Task 4 — Analisi predittiva: previsione picchi di feedback negativi

Il sistema già raccoglie, per ogni documento processato, sentiment label, score, entità, timestamp e sorgente. Con 6 mesi di storico questi dati diventano una serie temporale su cui è ragionevole costruire un modello che stimi la probabilità di un picco di negativi nelle prossime 24 ore.
L'output atteso è un segnale binario o una probabilità: *"nelle prossime 24 ore il volume di feedback negativi supererà una soglia critica?"*
## Feature engineering

Parto dai dati già presenti in `analysis_result` e `analysis_run`, aggregati su finestre temporali scorrevoli.

**Feature temporali (finestre 1h, 6h, 24h, 7d):**
- `negative_rate_Xh`: percentuale di feedback negativi nell'ultima X ore
- `negative_volume_Xh`: conteggio assoluto, separato dal rate perché un volume alto con rate basso è diverso da un volume basso con rate alto
- `avg_sentiment_score_Xh`: media degli score negativi, distingue feedback leggermente negativi da crisi vere
- `negative_rate_change`: derivata del rate tra finestre consecutive, cattura l'accelerazione prima ancora che il picco si manifesti

**Feature per sorgente e lingua:**
- `negative_rate_by_source`: email, app, web si comportano diversamente; un picco su `source-app` ha dinamiche diverse da `source-email`
- `language_distribution`: uno shift improvviso nella lingua può anticipare un evento geografico specifico

**Feature da NER (già estratte e in JSONB):**
- `entity_frequency_delta`: se un'entità compare molto più spesso del solito tra i negativi, è un segnale forte.
- `new_entities_in_negatives`: entità mai viste prima nei feedback negativi, potenziale indicatore di un nuovo problema


## Costruzione del dataset

L'unità di osservazione è una finestra temporale (es. ogni ora). Il target `y` è 1 se nelle 24 ore successive il volume di negativi supera la soglia (definibile come media + 2σ calcolata sullo storico). Le feature sono calcolate sulla finestra corrente e su quelle precedenti.

Con 6 mesi di dati orari si ottengono circa 4.000 osservazioni, sufficienti per modelli che generalizzano bene su serie temporali corte.

## Scelta del modello

Partirei da **XGBoost o LightGBM** su feature tabellari aggregate. I motivi concreti:

- gestiscono bene feature con distribuzione asimmetrica (i picchi sono eventi rari)
- l'importanza delle feature è interpretabile: utile per spiegare al team di supporto *perché* scatta l'allerta
- reggono dataset di questa dimensione senza richiedere GPU né infrastruttura complessa
- con `scale_pos_weight` o `class_weight` si bilancia facilmente il dataset sbilanciato (i picchi sono minoritari per definizione)

**Valutazione:** con dati temporali non si usa k-fold classico. Si usa walk-forward validation: si allena sui primi N mesi, si valuta sul mese N+1, si avanza. La metrica principale è il **recall sui positivi** (meglio un falso allarme che un picco mancato) affiancato da precision per non saturare il team con notifiche inutili.

Se i dati crescono e si vuole sfruttare la struttura sequenziale in modo più esplicito, si può valutare un **LSTM** o **Temporal Fusion Transformer**.

## Integrazione con il sistema esistente

Il modello si inserisce naturalmente a valle della pipeline esistente: ogni batch processato aggiorna le feature aggregate in una tabella dedicata (es. `feedback_hourly_stats`), un job schedulato ogni ora ricalcola la predizione e, se la probabilità supera la soglia configurabile, manda una notifica.
I campi già presenti nel DB (`sentiment_label`, `sentiment_score`, `entities`, `received_at`, `source_id`) coprono la maggior parte delle feature senza richiedere modifiche allo schema.
