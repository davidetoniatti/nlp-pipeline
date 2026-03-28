package pipeline

import (
	"fmt"
	"io"
	"log/slog"
	"testing"
	"time"
)

func TestRetryDelay(t *testing.T) {
	cfg := DefaultConfig()
	cfg.RetryBaseDelay = 100 * time.Millisecond
	cfg.MaxRetryDelay = 1 * time.Second
	cfg.RetryJitterFrac = 0 // Disable jitter for deterministic testing

	orch := &Orchestrator{cfg: cfg}

	tests := []struct {
		attempt  int
		expected time.Duration
	}{
		{0, 100 * time.Millisecond},
		{1, 200 * time.Millisecond},
		{2, 400 * time.Millisecond},
		{3, 800 * time.Millisecond},
		{4, 1000 * time.Millisecond}, // Clamped to MaxRetryDelay
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("Attempt %d", tt.attempt), func(t *testing.T) {
			got := orch.retryDelay(tt.attempt)
			if got != tt.expected {
				t.Errorf("retryDelay(%d) = %v; want %v", tt.attempt, got, tt.expected)
			}
		})
	}
}

func TestApplyResults_Robustness(t *testing.T) {
	orch := &Orchestrator{
		logger: DefaultConfigLogger(),
	}

	// Setup a batch with 3 documents
	batch := NewBatch("test-batch", 3)
	batch.IDs = []string{"doc-1", "doc-2", "doc-3"}
	for i := range batch.IDs {
		batch.Statuses[i] = StatusProcessing
	}

	// Simulate a response with:
	// 1. A valid result (doc-1)
	// 2. A duplicate result (doc-1 again)
	// 3. An unknown result (doc-99)
	// 4. A missing result (doc-2 and doc-3 are not in the response)
	resp := InferenceResponse{
		Results: []DocResult{
			{DocID: "doc-1", SentimentLabel: "Positive", SentimentScore: 0.9},
			{DocID: "doc-1", SentimentLabel: "Negative", SentimentScore: -0.1}, // Duplicate
			{DocID: "doc-99", SentimentLabel: "Neutral"},                        // Unknown
		},
		ModelVersions: ModelVersions{
			Sentiment: ModelMetadata{Name: "sent-v1"},
			NER:       ModelMetadata{Name: "ner-v1"},
			Summary:   ModelMetadata{Name: "sum-v1"},
		},
	}

	orch.applyResults(batch, resp)

	// doc-1 should be Done (from the first match)
	if batch.Statuses[0] != StatusDone {
		t.Errorf("doc-1 status = %v; want StatusDone", batch.Statuses[0])
	}

	// doc-2 and doc-3 should be Failed (missing from response)
	if batch.Statuses[1] != StatusFailed {
		t.Errorf("doc-2 status = %v; want StatusFailed", batch.Statuses[1])
	}
	if batch.Statuses[2] != StatusFailed {
		t.Errorf("doc-3 status = %v; want StatusFailed", batch.Statuses[2])
	}
}

// Helper for testing
func DefaultConfigLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}
