package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"sync"
	"syscall"
)

func loadMockData(filename string) ([]Message, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("open file failed: %w", err)
	}
	defer file.Close()

	var data []struct {
		ID       string `json:"ID"`
		SourceID string `json:"SourceID"`
		Text     string `json:"Text"`
	}

	if err := json.NewDecoder(file).Decode(&data); err != nil {
		return nil, fmt.Errorf("json decode failed: %w", err)
	}

	messages := make([]Message, len(data))
	for i, d := range data {
		messages[i] = Message{
			ID:       d.ID,
			SourceID: d.SourceID,
			Text:     d.Text,
		}
	}
	return messages, nil
}

func resolveMockDataPath() string {
	if envPath := os.Getenv("MOCK_DATA_FILE"); envPath != "" {
		return envPath
	}

	candidates := []string{
		"test_data.json",
		filepath.Join("..", "test_data.json"),
	}

	for _, candidate := range candidates {
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
	}

	return "test_data.json"
}

func statusString(s DocumentStatus) string {
	switch s {
	case StatusPending:
		return "pending"
	case StatusProcessing:
		return "processing"
	case StatusDone:
		return "done"
	case StatusFailed:
		return "failed"
	default:
		return "unknown"
	}
}

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	slog.SetDefault(logger)

	cfg := DefaultConfig()
	if envURL := os.Getenv("INFERENCE_URL"); envURL != "" {
		cfg.InferenceURL = envURL
	}

	databaseURL := os.Getenv("DATABASE_URL")
	if databaseURL == "" {
		logger.Error("DATABASE_URL is required")
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	store, err := NewStore(ctx, databaseURL, logger)
	if err != nil {
		logger.Error("failed to initialize store", "err", err)
		os.Exit(1)
	}
	defer store.Close()

	failureCfg := LoadFailureInjectionConfig()
	if failureCfg.Enabled {
		logger.Warn(
			"failure injection enabled",
			"fail_doc_ids_count", len(failureCfg.DocIDs),
			"fail_text_contains", failureCfg.TextContains,
		)
	}

	dataFile := resolveMockDataPath()

	logger.Info(
		"starting system",
		"inference_url", cfg.InferenceURL,
		"workers", cfg.NumWorkers,
		"batch_size", cfg.BatchSize,
		"inference_batch_size", cfg.InferenceBatchSize,
		"data_file", dataFile,
	)

	messages, err := loadMockData(dataFile)
	if err != nil {
		logger.Error("failed to load mock data", "err", err)
		os.Exit(1)
	}
	logger.Info("mock data loaded", "total_documents", len(messages))

	queue := NewMockQueue(messages)
	aiClient := NewHTTPInferenceClient(cfg.InferenceURL)

	resultsCh := make(chan ProcessedBatch, 100)
	orch := NewOrchestrator(cfg, queue, aiClient, resultsCh, logger)

	var persistWG sync.WaitGroup
	persistWG.Add(1)

	go func() {
		defer persistWG.Done()

		processedDocs := 0

		for item := range resultsCh {
			batch := item.Batch

			injected := applyFailureInjection(batch, failureCfg)
			if injected > 0 {
				logger.Warn(
					"failure injection applied",
					"batch_id", batch.ID,
					"injected_docs", injected,
				)
			}

			fmt.Printf("\n=== BATCH ANALYSIS: %s ===\n", batch.ID)
			fmt.Printf("Models: sentiment=%s | ner=%s | summary=%s | prompt=%s\n",
				batch.ModelVersions.SentimentModelID,
				batch.ModelVersions.NERModelID,
				batch.ModelVersions.SummaryModelID,
				batch.PromptVersion,
			)

			for i := 0; i < batch.Size; i++ {
				processedDocs++

				fmt.Printf("\n[%d] ID: %s\n", processedDocs, batch.IDs[i])
				fmt.Printf("  AnalysisRunID: %s\n", batch.AnalysisRunIDs[i])
				fmt.Printf("  SourceID: %s\n", batch.SourceIDs[i])
				fmt.Printf("  Text: %s\n", batch.Texts[i])
				fmt.Printf("  Status: %s (%d)\n", statusString(batch.Statuses[i]), batch.Statuses[i])
				fmt.Printf("  Sentiment: %s (%.2f)\n", batch.SentimentLabels[i], batch.SentimentScores[i])
				fmt.Printf("  ProcessingMs: %d\n", batch.ProcessingMs[i])

				if len(batch.Entities[i]) > 0 {
					fmt.Printf("  Entities: ")
					for _, e := range batch.Entities[i] {
						fmt.Printf("[%s: %s (%.2f)] ", e.Text, e.Label, e.Score)
					}
					fmt.Println()
				} else {
					fmt.Printf("  Entities: none detected\n")
				}

				if batch.Summaries[i] != "" {
					fmt.Printf("  Summary:\n    %s\n", batch.Summaries[i])
				}

				if batch.ProcessingErrors[i] != "" {
					fmt.Printf("  Error: %s\n", batch.ProcessingErrors[i])
				}
			}

			logger.Info("persisting batch", "batch_id", batch.ID, "size", batch.Size)

			if err := store.PersistBatch(ctx, batch); err != nil {
				logger.Error("batch persistence failed", "batch_id", batch.ID, "err", err)
				if nackErr := item.Nack(); nackErr != nil {
					logger.Error("batch nack failed", "batch_id", batch.ID, "err", nackErr)
				} else {
					logger.Warn("batch not persisted, sent for retry", "batch_id", batch.ID)
				}
				continue
			}

			stats, err := store.VerifyBatchPersistence(ctx, batch)
			if err != nil {
				logger.Error("batch verification failed", "batch_id", batch.ID, "err", err)
				if nackErr := item.Nack(); nackErr != nil {
					logger.Error("batch nack failed after verification error", "batch_id", batch.ID, "err", nackErr)
				}
				continue
			}

			expectedDone := countBatchStatus(batch, StatusDone)
			expectedFailed := countBatchStatus(batch, StatusFailed)

			if stats.RunsTotal != batch.Size ||
				stats.RunsDone != expectedDone ||
				stats.RunsFailed != expectedFailed ||
				stats.ResultsTotal != expectedDone {
				logger.Error(
					"batch verification mismatch",
					"batch_id", batch.ID,
					"expected_runs_total", batch.Size,
					"actual_runs_total", stats.RunsTotal,
					"expected_done", expectedDone,
					"actual_done", stats.RunsDone,
					"expected_failed", expectedFailed,
					"actual_failed", stats.RunsFailed,
					"expected_results", expectedDone,
					"actual_results", stats.ResultsTotal,
				)
				if nackErr := item.Nack(); nackErr != nil {
					logger.Error("batch nack failed after verification mismatch", "batch_id", batch.ID, "err", nackErr)
				}
				continue
			}

			logger.Info(
				"batch persistence verified",
				"batch_id", batch.ID,
				"runs_total", stats.RunsTotal,
				"runs_done", stats.RunsDone,
				"runs_failed", stats.RunsFailed,
				"results_total", stats.ResultsTotal,
			)

			if err := item.Ack(); err != nil {
				logger.Error("batch ack failed", "batch_id", batch.ID, "err", err)
			} else {
				logger.Info(
					"batch persisted and acked",
					"batch_id", batch.ID,
					"total_processed", processedDocs,
				)
			}
		}
	}()

	logger.Info("orchestrator running")

	if err := orch.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
		logger.Error("orchestrator terminated with error", "err", err)
		close(resultsCh)
		persistWG.Wait()
		os.Exit(1)
	}

	close(resultsCh)
	persistWG.Wait()
	logger.Info("shutdown completed successfully")
}
