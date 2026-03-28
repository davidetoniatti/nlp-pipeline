package pipeline

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/google/uuid"
	"io"
	"log/slog"
	"math"
	"math/rand"
	"net/http"
	"sync"
	"time"
)

/* Message is the input unit consumed by the orchestrator. */
type Message struct {
	ID       string
	SourceID string
	Lang     string
	Text     string
}

/*
QueueConsumer abstracts the input queue.
The orchestrator does not care if messages come from memory,
Kafka, SQS, or anywhere else.
*/
type QueueConsumer interface {
	Consume(ctx context.Context) <-chan Message
	Ack(msgID string) error
	Nack(msgID string) error
}

/* InferenceClient is the interface to the NLP service. */
type InferenceClient interface {
	Analyze(ctx context.Context, req InferenceRequest) (InferenceResponse, error)
}

/* InferenceRequest is the payload sent to the NLP service. */
type InferenceRequest struct {
	DocIDs    []string `json:"doc_ids"`
	Texts     []string `json:"texts"`
	BatchSize int      `json:"batch_size"`
}

/* InferenceResponse is the batch-level reply from the NLP service. */
type InferenceResponse struct {
	Results       []DocResult   `json:"results"`
	ModelVersions ModelVersions `json:"model_versions"`
}

/*
ModelVersions carries model metadata returned by the NLP service.
*/
type ModelVersions struct {
	Sentiment     ModelMetadata `json:"sentiment"`
	NER           ModelMetadata `json:"ner"`
	Summary       ModelMetadata `json:"summary"`
	PromptVersion string        `json:"prompt_version"`
	PromptHash    string        `json:"prompt_hash"`
	ServiceGitSHA string        `json:"service_git_sha"`
}

/* ModelMetadata carries details for a specific model. */
type ModelMetadata struct {
	Name      string `json:"name"`
	Version   string `json:"version"`
	Revision  string `json:"revision"`
	Tokenizer string `json:"tokenizer"`
	Provider  string `json:"provider"`
}

/* DocResult is the per-document inference output. */
type DocResult struct {
	DocID          string      `json:"doc_id"`
	SentimentScore float32     `json:"sentiment_score"`
	SentimentLabel string      `json:"sentiment_label"`
	Entities       []EntityDTO `json:"entities"`
	Summary        string      `json:"summary,omitempty"`
	Error          string      `json:"error,omitempty"`
}

/* EntityDTO is the wire format used by the NLP service. */
type EntityDTO struct {
	Text  string  `json:"text"`
	Label string  `json:"label"`
	Start int     `json:"start"`
	End   int     `json:"end"`
	Score float32 `json:"score"`
}

/*
HTTPStatusError preserves the upstream HTTP status code.
This lets the retry logic distinguish retriable failures
from client-side errors that should fail fast.
*/
type HTTPStatusError struct {
	StatusCode int
	Body       string
}

/* Error returns a compact error string for logs and retries. */
func (e *HTTPStatusError) Error() string {
	return fmt.Sprintf("inference service returned %d: %s", e.StatusCode, e.Body)
}

/* OrchestratorConfig groups the runtime knobs of the pipeline. */
type OrchestratorConfig struct {
	NumWorkers         int
	BatchSize          int
	InferenceBatchSize int
	WorkerTimeout      time.Duration
	MaxRetries         int
	RetryBaseDelay     time.Duration
	MaxRetryDelay      time.Duration
	RetryJitterFrac    float64
	InferenceURL       string
	FlushInterval      time.Duration
}

/* DefaultConfig returns conservative defaults for local runs. */
func DefaultConfig() OrchestratorConfig {
	return OrchestratorConfig{
		NumWorkers:         8,
		BatchSize:          32,
		InferenceBatchSize: 32,
		WorkerTimeout:      120 * time.Second,
		MaxRetries:         3,
		RetryBaseDelay:     200 * time.Millisecond,
		MaxRetryDelay:      5 * time.Second,
		RetryJitterFrac:    0.20,
		InferenceURL:       "http://localhost:8080/analyze",
		FlushInterval:      500 * time.Millisecond,
	}
}

/*
HTTPInferenceClient is the default HTTP implementation of InferenceClient.
Timeouts are driven by request contexts, not by the client itself.
*/
type HTTPInferenceClient struct {
	client  *http.Client
	baseURL string
}

/* NewHTTPInferenceClient builds an HTTP client tuned for reuse. */
func NewHTTPInferenceClient(baseURL string) *HTTPInferenceClient {
	return &HTTPInferenceClient{
		baseURL: baseURL,
		client: &http.Client{
			/* Request lifetime is controlled by the caller's context. */
			Timeout: 0,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 20,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}
}

/*
Analyze sends one batch to the NLP service and decodes the reply.
Non-200 responses are preserved as HTTPStatusError so callers
can make retry decisions based on status code.
*/
func (c *HTTPInferenceClient) Analyze(ctx context.Context, req InferenceRequest) (InferenceResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return InferenceResponse{}, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL, bytes.NewReader(body))
	if err != nil {
		return InferenceResponse{}, fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return InferenceResponse{}, fmt.Errorf("http call: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return InferenceResponse{}, &HTTPStatusError{
			StatusCode: resp.StatusCode,
			Body:       string(raw),
		}
	}

	var result InferenceResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return InferenceResponse{}, fmt.Errorf("decode response: %w", err)
	}
	return result, nil
}

/*
ProcessedBatch is the handoff between workers and persistence.
The batch is ready, but message acking is delayed until persistence succeeds.
*/
type ProcessedBatch struct {
	Batch *Batch
	Ack   func() error
	Nack  func() error
}

/*
Orchestrator glues queue consumption, batching, inference, and worker fan-out.
Persistence is optional. If results is nil, workers ack directly.
*/
type Orchestrator struct {
	cfg     OrchestratorConfig
	queue   QueueConsumer
	ai      InferenceClient
	results chan<- ProcessedBatch
	logger  *slog.Logger
	wg      sync.WaitGroup
}

/* NewOrchestrator wires the pipeline components together. */
func NewOrchestrator(
	cfg OrchestratorConfig,
	queue QueueConsumer,
	ai InferenceClient,
	results chan<- ProcessedBatch,
	logger *slog.Logger,
) *Orchestrator {
	return &Orchestrator{
		cfg:     cfg,
		queue:   queue,
		ai:      ai,
		results: results,
		logger:  logger,
	}
}

/*
Run starts the workers, feeds them with batches, and waits for shutdown.
The input side stops first, then workers drain the remaining batches.
*/
func (o *Orchestrator) Run(ctx context.Context) error {
	batchCh := make(chan *Batch, o.cfg.NumWorkers)

	for i := 0; i < o.cfg.NumWorkers; i++ {
		o.wg.Add(1)
		go o.runWorker(ctx, i, batchCh)
	}

	err := o.accumulate(ctx, batchCh)
	close(batchCh)
	o.wg.Wait()

	return err
}

/*
accumulate reads messages and groups them into batches.
A batch is flushed either when it is full or when the flush timer fires.
*/
func (o *Orchestrator) accumulate(ctx context.Context, batchCh chan<- *Batch) error {
	msgCh := o.queue.Consume(ctx)

	var (
		pending  []Message
		batchSeq int
	)

	flush := func() {
		if len(pending) == 0 {
			return
		}

		batchSeq++
		b := NewBatch(fmt.Sprintf("batch-%05d", batchSeq), len(pending))
		for i, msg := range pending {
			b.SetDocumentWithLangAndHash(i, msg.ID, msg.SourceID, msg.Lang, msg.Text, "")
		}

		/* Reuse the slice storage after the batch has been materialized. */
		pending = pending[:0]

		select {
		case batchCh <- b:
		case <-ctx.Done():
		}
	}

	ticker := time.NewTicker(o.cfg.FlushInterval)
	defer ticker.Stop()

	for {
		select {
		case msg, ok := <-msgCh:
			if !ok {
				flush()
				return nil
			}

			pending = append(pending, msg)
			if len(pending) >= o.cfg.BatchSize {
				flush()
			}

		case <-ticker.C:
			flush()

		case <-ctx.Done():
			flush()
			return ctx.Err()
		}
	}
}

/*
runWorker processes batches one by one.
If a persistence stage exists, ack/nack is deferred to the consumer
of ProcessedBatch. Otherwise the worker acks directly.
*/
func (o *Orchestrator) runWorker(ctx context.Context, id int, batchCh <-chan *Batch) {
	defer o.wg.Done()
	log := o.logger.With("worker_id", id)

	for batch := range batchCh {
		log.Info("processing batch", "batch_id", batch.ID, "size", batch.Size)

		err := o.processBatchWithRetry(ctx, batch)
		if err != nil {
			log.Error("batch failed definitively", "batch_id", batch.ID, "err", err)
		}

		ackFn := func() error {
			for _, msgID := range batch.IDs {
				if err := o.queue.Ack(msgID); err != nil {
					return fmt.Errorf("ack msg %s: %w", msgID, err)
				}
			}
			return nil
		}

		nackFn := func() error {
			var lastErr error
			for _, msgID := range batch.IDs {
				if err := o.queue.Nack(msgID); err != nil {
					lastErr = fmt.Errorf("nack msg %s: %w", msgID, err)
				}
			}
			return lastErr
		}

		if o.results != nil {
			select {
			case o.results <- ProcessedBatch{
				Batch: batch,
				Ack:   ackFn,
				Nack:  nackFn,
			}:
			case <-ctx.Done():
				return
			}
			continue
		}

		if err != nil {
			if nackErr := nackFn(); nackErr != nil {
				log.Error("nack batch failed", "batch_id", batch.ID, "err", nackErr)
			}
			continue
		}

		if ackErr := ackFn(); ackErr != nil {
			log.Error("ack batch failed", "batch_id", batch.ID, "err", ackErr)
		}
	}
}

/* newAnalysisRunID returns a fresh logical run identifier. */
func newAnalysisRunID() string {
	return uuid.NewString()
}

/*
ensureAnalysisRunIDs assigns one analysis run id per document.
Transport retries reuse the same logical run id. Without this,
retries would look like new analysis runs instead of the same run retried.
*/
func (o *Orchestrator) ensureAnalysisRunIDs(batch *Batch) {
	for i := 0; i < batch.Size; i++ {
		if batch.AnalysisRunIDs[i] == "" {
			batch.SetAnalysisRunID(i, newAnalysisRunID())
		}
	}
}

/*
processBatchWithRetry sends one batch to the NLP service with retry logic.
Only retriable failures are retried. Client-side HTTP errors fail fast,
except 429 which is treated as temporary backpressure.
*/
func (o *Orchestrator) processBatchWithRetry(ctx context.Context, batch *Batch) error {
	/* Assign run ids once so retries stay part of the same logical run. */
	o.ensureAnalysisRunIDs(batch)

	req := InferenceRequest{
		DocIDs:    batch.IDs,
		Texts:     batch.Texts,
		BatchSize: o.cfg.InferenceBatchSize,
	}

	var lastErr error
	for attempt := 0; attempt <= o.cfg.MaxRetries; attempt++ {
		if attempt > 0 {
			delay := o.retryDelay(attempt - 1)
			o.logger.Info(
				"retrying batch",
				"batch_id", batch.ID,
				"attempt", attempt,
				"delay", delay,
				"last_err", lastErr,
			)

			select {
			case <-time.After(delay):
			case <-ctx.Done():
				return ctx.Err()
			}
		}

		/*
			Mark documents right before the remote call.
			This keeps ProcessingMs tied to actual work time,
			not to time spent sleeping between retries.
		*/
		for i := 0; i < batch.Size; i++ {
			batch.MarkProcessing(i)
		}

		callCtx, cancel := context.WithTimeout(ctx, o.cfg.WorkerTimeout)
		resp, err := o.ai.Analyze(callCtx, req)
		cancel()

		if err == nil {
			o.applyResults(batch, resp)
			return nil
		}
		lastErr = err

		if ctx.Err() != nil {
			return ctx.Err()
		}

		var httpErr *HTTPStatusError
		if errors.As(err, &httpErr) {
			if httpErr.StatusCode >= 400 &&
				httpErr.StatusCode < 500 &&
				httpErr.StatusCode != http.StatusTooManyRequests {
				return err
			}
		}
	}

	/* If we get here, every attempt failed. */
	for i := 0; i < batch.Size; i++ {
		if batch.Statuses[i] != StatusDone {
			batch.SetError(i, fmt.Errorf("all attempts failed: %w", lastErr))
		}
	}

	return fmt.Errorf("all attempts failed: %w", lastErr)
}

/*
retryDelay computes exponential backoff with optional positive jitter.
Jitter spreads retries over time and avoids synchronized retry storms.
*/
func (o *Orchestrator) retryDelay(attempt int) time.Duration {
	base := float64(o.cfg.RetryBaseDelay) * math.Pow(2, float64(attempt))
	delay := time.Duration(base)

	if o.cfg.MaxRetryDelay > 0 && delay > o.cfg.MaxRetryDelay {
		delay = o.cfg.MaxRetryDelay
	}

	if o.cfg.RetryJitterFrac <= 0 {
		return delay
	}

	maxJitter := float64(delay) * o.cfg.RetryJitterFrac
	jitter := time.Duration(rand.Float64() * maxJitter)
	return delay + jitter
}

/* dtoToEntities converts wire DTOs into the internal batch representation. */
func dtoToEntities(dtos []EntityDTO) []Entity {
	out := make([]Entity, len(dtos))
	for i, d := range dtos {
		out[i] = Entity{
			Text:  d.Text,
			Label: d.Label,
			Start: int32(d.Start),
			End:   int32(d.End),
			Score: d.Score,
		}
	}
	return out
}

/*
applyResults matches inference results back to batch slots by document id.
Matching by id instead of position is more defensive and survives
reordered, missing, duplicate, or unknown results.
*/
func (o *Orchestrator) applyResults(batch *Batch, resp InferenceResponse) {
	if len(resp.Results) != batch.Size {
		o.logger.Warn(
			"response size mismatch",
			"batch_id", batch.ID,
			"expected", batch.Size,
			"got", len(resp.Results),
		)
	}

	idxByID := make(map[string]int, batch.Size)
	seen := make([]bool, batch.Size)

	for i, id := range batch.IDs {
		idxByID[id] = i
	}

	for _, r := range resp.Results {
		idx, ok := idxByID[r.DocID]
		if !ok {
			o.logger.Warn("unknown doc_id in response", "doc_id", r.DocID, "batch_id", batch.ID)
			continue
		}
		if seen[idx] {
			o.logger.Warn("duplicate doc_id in response", "doc_id", r.DocID, "batch_id", batch.ID)
			continue
		}

		seen[idx] = true

		if r.Error != "" {
			batch.SetError(idx, fmt.Errorf("inference error: %s", r.Error))
			continue
		}

		batch.SetResult(idx, r.SentimentScore, r.SentimentLabel, dtoToEntities(r.Entities), r.Summary)
	}

	/* Any document not seen in the reply is treated as failed. */
	for i := 0; i < batch.Size; i++ {
		if !seen[i] {
			batch.SetError(i, fmt.Errorf("missing result for doc_id=%s", batch.IDs[i]))
		}
	}

	/*
		Model versions coming from Python are logical names, not DB ids.
		The persistence layer can upsert MODEL_VERSION rows
		and resolve real foreign keys later.
	*/
	batch.ModelVersions = BatchModelVersions{
		Sentiment: ModelMetadataDTO{
			Name:      resp.ModelVersions.Sentiment.Name,
			Version:   resp.ModelVersions.Sentiment.Version,
			Revision:  resp.ModelVersions.Sentiment.Revision,
			Tokenizer: resp.ModelVersions.Sentiment.Tokenizer,
			Provider:  resp.ModelVersions.Sentiment.Provider,
		},
		NER: ModelMetadataDTO{
			Name:      resp.ModelVersions.NER.Name,
			Version:   resp.ModelVersions.NER.Version,
			Revision:  resp.ModelVersions.NER.Revision,
			Tokenizer: resp.ModelVersions.NER.Tokenizer,
			Provider:  resp.ModelVersions.NER.Provider,
		},
		Summary: ModelMetadataDTO{
			Name:      resp.ModelVersions.Summary.Name,
			Version:   resp.ModelVersions.Summary.Version,
			Revision:  resp.ModelVersions.Summary.Revision,
			Tokenizer: resp.ModelVersions.Summary.Tokenizer,
			Provider:  resp.ModelVersions.Summary.Provider,
		},
	}
	batch.PromptVersion = resp.ModelVersions.PromptVersion
	batch.PromptHash = resp.ModelVersions.PromptHash
	batch.ServiceGitSHA = resp.ModelVersions.ServiceGitSHA

	o.logger.Info(
		"applied batch results",
		"batch_id", batch.ID,
		"prompt_version", resp.ModelVersions.PromptVersion,
		"service_git_sha", resp.ModelVersions.ServiceGitSHA,
	)
}

/* MockQueue is an in-memory queue used by the prototype. */
type MockQueue struct {
	messages []Message
}

/* NewMockQueue wraps a fixed slice of messages as a queue. */
func NewMockQueue(messages []Message) *MockQueue {
	return &MockQueue{messages: messages}
}

/*
Consume streams all mock messages and then closes the channel.
This behaves like a finite queue, good enough for local runs and demos.
*/
func (q *MockQueue) Consume(ctx context.Context) <-chan Message {
	ch := make(chan Message, len(q.messages))
	go func() {
		defer close(ch)
		for _, m := range q.messages {
			select {
			case ch <- m:
			case <-ctx.Done():
				return
			}
		}
	}()
	return ch
}

/* Ack is a no-op in the mock queue. */
func (q *MockQueue) Ack(msgID string) error { return nil }

/* Nack is a no-op in the mock queue. */
func (q *MockQueue) Nack(msgID string) error { return nil }
