package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"math/rand"
	"net/http"
	"sync"
	"time"
)

type Message struct {
	ID       string
	SourceID string
	Text     string
}

type QueueConsumer interface {
	Consume(ctx context.Context) <-chan Message
	Ack(msgID string) error
	Nack(msgID string) error
}

type InferenceClient interface {
	Analyze(ctx context.Context, req InferenceRequest) (InferenceResponse, error)
}

type InferenceRequest struct {
	DocIDs    []string `json:"doc_ids"`
	Texts     []string `json:"texts"`
	BatchSize int      `json:"batch_size"`
}

type InferenceResponse struct {
	Results       []DocResult   `json:"results"`
	ModelVersions ModelVersions `json:"model_versions"`
}

type ModelVersions struct {
	SentimentModel string `json:"sentiment_model"`
	NERModel       string `json:"ner_model"`
	SummaryModel   string `json:"summary_model"`
	PromptVersion  string `json:"prompt_version"`
}

type DocResult struct {
	DocID          string      `json:"doc_id"`
	SentimentScore float32     `json:"sentiment_score"`
	SentimentLabel string      `json:"sentiment_label"`
	Entities       []EntityDTO `json:"entities"`
	Summary        string      `json:"summary,omitempty"`
	Error          string      `json:"error,omitempty"`
}

type EntityDTO struct {
	Text  string  `json:"text"`
	Label string  `json:"label"`
	Start int     `json:"start"`
	End   int     `json:"end"`
	Score float32 `json:"score"`
}

type HTTPStatusError struct {
	StatusCode int
	Body       string
}

func (e *HTTPStatusError) Error() string {
	return fmt.Sprintf("inference service returned %d: %s", e.StatusCode, e.Body)
}

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

type HTTPInferenceClient struct {
	client  *http.Client
	baseURL string
}

func NewHTTPInferenceClient(baseURL string) *HTTPInferenceClient {
	return &HTTPInferenceClient{
		baseURL: baseURL,
		client: &http.Client{
			// Timeout is controlled via request context in Analyze.
			Timeout: 0,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 20,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}
}

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

type ProcessedBatch struct {
	Batch *Batch
	Ack   func() error
	Nack  func() error
}

type Orchestrator struct {
	cfg     OrchestratorConfig
	queue   QueueConsumer
	ai      InferenceClient
	results chan<- ProcessedBatch
	logger  *slog.Logger
	wg      sync.WaitGroup
}

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
			b.SetDocument(i, msg.ID, msg.SourceID, msg.Text)
		}

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

func (o *Orchestrator) runWorker(ctx context.Context, id int, batchCh <-chan *Batch) {
	defer o.wg.Done()
	log := o.logger.With("worker_id", id)

	for batch := range batchCh {
		log.Info("processing batch", "batch_id", batch.ID, "size", batch.Size)

		err := o.processBatchWithRetry(ctx, batch)
		if err != nil {
			log.Error("batch failed definitively", "batch_id", batch.ID, "err", err)
			for _, msgID := range batch.IDs {
				if nackErr := o.queue.Nack(msgID); nackErr != nil {
					log.Error("nack failed", "msg_id", msgID, "err", nackErr)
				}
			}
			continue
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

		if err := ackFn(); err != nil {
			log.Error("ack batch failed", "batch_id", batch.ID, "err", err)
		}
	}
}

func newAnalysisRunID(batchID string, docIndex int) string {
	return fmt.Sprintf(
		"%s-run-%02d-%d-%d",
		batchID,
		docIndex,
		time.Now().UTC().UnixNano(),
		rand.Uint64(),
	)
}

// ensureAnalysisRunIDs assigns one ANALYSIS_RUN id per document.
//
// Technical retries reuse the same run id so that transport retries do not
// create duplicate logical runs for the same document.
func (o *Orchestrator) ensureAnalysisRunIDs(batch *Batch) {
	for i := 0; i < batch.Size; i++ {
		if batch.AnalysisRunIDs[i] == "" {
			batch.SetAnalysisRunID(i, newAnalysisRunID(batch.ID, i))
		}
	}
}

func (o *Orchestrator) processBatchWithRetry(ctx context.Context, batch *Batch) error {
	// Assign one run id per document before the first attempt.
	// Retries reuse the same run ids.
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

		// Mark documents as processing immediately before the remote inference
		// call so ProcessingMs measures execution time rather than backoff time.
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
			if httpErr.StatusCode >= 400 && httpErr.StatusCode < 500 && httpErr.StatusCode != http.StatusTooManyRequests {
				return err
			}
		}
	}

	for i := 0; i < batch.Size; i++ {
		if batch.Statuses[i] != StatusDone {
			batch.SetError(i, fmt.Errorf("all attempts failed: %w", lastErr))
		}
	}

	return fmt.Errorf("all attempts failed: %w", lastErr)
}

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

	for i := 0; i < batch.Size; i++ {
		if !seen[i] {
			batch.SetError(i, fmt.Errorf("missing result for doc_id=%s", batch.IDs[i]))
		}
	}

	// NOTE: the Python service returns model names / references, not DB UUIDs.
	// At persistence time you can upsert MODEL_VERSION and resolve real FK ids.
	batch.ModelVersions = BatchModelVersions{
		SentimentModelID: resp.ModelVersions.SentimentModel,
		NERModelID:       resp.ModelVersions.NERModel,
		SummaryModelID:   resp.ModelVersions.SummaryModel,
	}
	batch.PromptVersion = resp.ModelVersions.PromptVersion

	o.logger.Info(
		"applied batch results",
		"batch_id", batch.ID,
		"prompt_version", resp.ModelVersions.PromptVersion,
	)
}

type MockQueue struct {
	messages []Message
}

func NewMockQueue(messages []Message) *MockQueue {
	return &MockQueue{messages: messages}
}

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

func (q *MockQueue) Ack(msgID string) error  { return nil }
func (q *MockQueue) Nack(msgID string) error { return nil }
