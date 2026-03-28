//go:build integration

package pipeline

import (
	"context"
	"os"
	"testing"
	"time"
)

func TestStore_PersistAndVerify(t *testing.T) {
	databaseURL := os.Getenv("DATABASE_URL")
	if databaseURL == "" {
		t.Skip("DATABASE_URL not set, skipping integration test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	logger := DefaultConfigLogger()
	store, err := NewStore(ctx, databaseURL, logger)
	if err != nil {
		t.Fatalf("Failed to create store: %v", err)
	}
	defer store.Close()

	// Create a batch with one successful and one failed document
	batch := NewBatch("int-test-batch", 2)
	batch.IDs[0] = "doc-ok"
	batch.SourceIDs[0] = "src-1"
	batch.Texts[0] = "This is a successful document."
	batch.AnalysisRunIDs[0] = "run-ok"
	batch.Statuses[0] = StatusDone
	batch.SentimentLabels[0] = "Positive"
	batch.SentimentScores[0] = 0.95
	batch.ProcessingMs[0] = 120

	batch.IDs[1] = "doc-err"
	batch.SourceIDs[1] = "src-1"
	batch.Texts[1] = "This is a failed document."
	batch.AnalysisRunIDs[1] = "run-err"
	batch.Statuses[1] = StatusFailed
	batch.ProcessingErrors[1] = "something went wrong"
	batch.ProcessingMs[1] = 45

	batch.ModelVersions = BatchModelVersions{
		Sentiment: ModelMetadataDTO{Name: "sent-v1", Provider: "hf"},
		NER:       ModelMetadataDTO{Name: "ner-v1", Provider: "hf"},
		Summary:   ModelMetadataDTO{Name: "sum-v1", Provider: "groq"},
	}
	batch.ServiceGitSHA = "test-sha"

	// Persist the batch
	if err := store.PersistBatch(ctx, batch); err != nil {
		t.Fatalf("PersistBatch failed: %v", err)
	}

	// Verify the batch
	stats, err := store.VerifyBatchPersistence(ctx, batch)
	if err != nil {
		t.Fatalf("VerifyBatchPersistence failed: %v", err)
	}

	if stats.RunsTotal != 2 {
		t.Errorf("stats.RunsTotal = %d; want 2", stats.RunsTotal)
	}
	if stats.RunsDone != 1 {
		t.Errorf("stats.RunsDone = %d; want 1", stats.RunsDone)
	}
	if stats.RunsFailed != 1 {
		t.Errorf("stats.RunsFailed = %d; want 1", stats.RunsFailed)
	}
	if stats.ResultsTotal != 1 {
		t.Errorf("stats.ResultsTotal = %d; want 1", stats.ResultsTotal)
	}
}
