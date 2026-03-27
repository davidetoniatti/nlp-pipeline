package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Store struct {
	pool   *pgxpool.Pool
	logger *slog.Logger
}

func NewStore(ctx context.Context, databaseURL string, logger *slog.Logger) (*Store, error) {
	cfg, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		return nil, fmt.Errorf("parse database url: %w", err)
	}

	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("create db pool: %w", err)
	}

	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("ping db: %w", err)
	}

	return &Store{
		pool:   pool,
		logger: logger,
	}, nil
}

func (s *Store) Close() {
	if s.pool != nil {
		s.pool.Close()
	}
}

func (s *Store) PersistBatch(ctx context.Context, batch *Batch) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}

	committed := false
	defer func() {
		if !committed {
			_ = tx.Rollback(ctx)
		}
	}()

	sentimentModelID, err := s.upsertModelVersion(ctx, tx, batch.ModelVersions.SentimentModelID, "sentiment")
	if err != nil {
		return fmt.Errorf("upsert sentiment model: %w", err)
	}

	nerModelID, err := s.upsertModelVersion(ctx, tx, batch.ModelVersions.NERModelID, "ner")
	if err != nil {
		return fmt.Errorf("upsert ner model: %w", err)
	}

	summaryModelID, err := s.upsertModelVersion(ctx, tx, batch.ModelVersions.SummaryModelID, "summary")
	if err != nil {
		return fmt.Errorf("upsert summary model: %w", err)
	}

	for i := 0; i < batch.Size; i++ {
		sourceUUID := normalizeUUID(batch.SourceIDs[i], "source:"+batch.SourceIDs[i])
		documentUUID := normalizeUUID(batch.IDs[i], "document:"+batch.IDs[i])

		if err := s.upsertSourceMetadata(ctx, tx, sourceUUID, batch.SourceIDs[i]); err != nil {
			return fmt.Errorf("upsert source_metadata for doc %s: %w", batch.IDs[i], err)
		}

		if err := s.upsertDocument(ctx, tx, batch, i, documentUUID, sourceUUID); err != nil {
			return fmt.Errorf("upsert document %s: %w", batch.IDs[i], err)
		}

		startedAt, completedAt := batchTimestamps(batch, i)

		if err := s.upsertAnalysisRun(
			ctx,
			tx,
			batch,
			i,
			documentUUID,
			sentimentModelID,
			nerModelID,
			summaryModelID,
			startedAt,
			completedAt,
		); err != nil {
			return fmt.Errorf("upsert analysis_run %s: %w", batch.AnalysisRunIDs[i], err)
		}

		if batch.Statuses[i] == StatusDone {
			if err := s.upsertAnalysisResult(ctx, tx, batch, i); err != nil {
				return fmt.Errorf("upsert analysis_result for run %s: %w", batch.AnalysisRunIDs[i], err)
			}
		}
	}

	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit tx: %w", err)
	}
	committed = true

	s.logger.Info("batch persisted to postgres", "batch_id", batch.ID, "size", batch.Size)
	return nil
}

func (s *Store) upsertSourceMetadata(ctx context.Context, tx pgxTx, sourceUUID uuid.UUID, rawSourceID string) error {
	name := "source-" + sanitizeLabel(rawSourceID)

	_, err := tx.Exec(ctx, `
		INSERT INTO source_metadata (
			id,
			name,
			platform_type,
			url,
			default_language
		)
		VALUES ($1, $2, $3, $4, $5)
		ON CONFLICT (id) DO UPDATE
		SET
			name = EXCLUDED.name,
			platform_type = EXCLUDED.platform_type
	`,
		sourceUUID,
		name,
		"mock",
		nil,
		"unknown",
	)

	return err
}

func (s *Store) upsertDocument(
	ctx context.Context,
	tx pgxTx,
	batch *Batch,
	i int,
	documentUUID uuid.UUID,
	sourceUUID uuid.UUID,
) error {
	var hashValue any
	if strings.TrimSpace(batch.Hashes[i]) != "" {
		hashValue = batch.Hashes[i]
	}

	_, err := tx.Exec(ctx, `
		INSERT INTO document (
			id,
			source_id,
			original_text,
			hash,
			language,
			status,
			received_at
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
		ON CONFLICT (id) DO UPDATE
		SET
			source_id = EXCLUDED.source_id,
			original_text = EXCLUDED.original_text,
			hash = EXCLUDED.hash,
			status = EXCLUDED.status
	`,
		documentUUID,
		sourceUUID,
		batch.Texts[i],
		hashValue,
		nil,
		statusString(batch.Statuses[i]),
		batch.CreatedAt,
	)

	return err
}

func (s *Store) upsertModelVersion(ctx context.Context, tx pgxTx, modelName, taskType string) (uuid.UUID, error) {
	modelName = strings.TrimSpace(modelName)
	if modelName == "" {
		modelName = "unknown"
	}

	provider := providerForTask(taskType)

	var id uuid.UUID
	err := tx.QueryRow(ctx, `
		INSERT INTO model_version (
			model_name,
			version,
			task_type,
			provider,
			artifact_uri
		)
		VALUES ($1, $2, $3, $4, $5)
		ON CONFLICT (model_name, version, task_type, provider) DO UPDATE
		SET artifact_uri = EXCLUDED.artifact_uri
		RETURNING id
	`,
		modelName,
		"unknown",
		taskType,
		provider,
		modelName,
	).Scan(&id)

	if err != nil {
		return uuid.Nil, err
	}

	return id, nil
}

func (s *Store) upsertAnalysisRun(
	ctx context.Context,
	tx pgxTx,
	batch *Batch,
	i int,
	documentUUID uuid.UUID,
	sentimentModelID uuid.UUID,
	nerModelID uuid.UUID,
	summaryModelID uuid.UUID,
	startedAt *time.Time,
	completedAt *time.Time,
) error {
	var promptVersion any
	if strings.TrimSpace(batch.PromptVersion) != "" {
		promptVersion = batch.PromptVersion
	}

	var errorMessage any
	if strings.TrimSpace(batch.ProcessingErrors[i]) != "" {
		errorMessage = batch.ProcessingErrors[i]
	}

	_, err := tx.Exec(ctx, `
		INSERT INTO analysis_run (
			id,
			document_id,
			sentiment_model_id,
			ner_model_id,
			summary_model_id,
			run_status,
			trigger_type,
			prompt_version,
			started_at,
			completed_at,
			error_message
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
		ON CONFLICT (id) DO UPDATE
		SET
			document_id = EXCLUDED.document_id,
			sentiment_model_id = EXCLUDED.sentiment_model_id,
			ner_model_id = EXCLUDED.ner_model_id,
			summary_model_id = EXCLUDED.summary_model_id,
			run_status = EXCLUDED.run_status,
			trigger_type = EXCLUDED.trigger_type,
			prompt_version = EXCLUDED.prompt_version,
			started_at = EXCLUDED.started_at,
			completed_at = EXCLUDED.completed_at,
			error_message = EXCLUDED.error_message
	`,
		batch.AnalysisRunIDs[i],
		documentUUID,
		sentimentModelID,
		nerModelID,
		summaryModelID,
		statusString(batch.Statuses[i]),
		"batch_pipeline",
		promptVersion,
		startedAt,
		completedAt,
		errorMessage,
	)

	return err
}

func (s *Store) upsertAnalysisResult(ctx context.Context, tx pgxTx, batch *Batch, i int) error {
	entitiesJSON, err := json.Marshal(batch.Entities[i])
	if err != nil {
		return fmt.Errorf("marshal entities: %w", err)
	}

	var summaryValue any
	if strings.TrimSpace(batch.Summaries[i]) != "" {
		summaryValue = batch.Summaries[i]
	}

	_, err = tx.Exec(ctx, `
		INSERT INTO analysis_result (
			analysis_run_id,
			sentiment_score,
			sentiment_label,
			entities,
			summary,
			processing_ms
		)
		VALUES ($1, $2, $3, $4::jsonb, $5, $6)
		ON CONFLICT (analysis_run_id) DO UPDATE
		SET
			sentiment_score = EXCLUDED.sentiment_score,
			sentiment_label = EXCLUDED.sentiment_label,
			entities = EXCLUDED.entities,
			summary = EXCLUDED.summary,
			processing_ms = EXCLUDED.processing_ms
	`,
		batch.AnalysisRunIDs[i],
		batch.SentimentScores[i],
		batch.SentimentLabels[i],
		string(entitiesJSON),
		summaryValue,
		batch.ProcessingMs[i],
	)

	return err
}

type pgxTx interface {
	Exec(ctx context.Context, sql string, arguments ...any) (pgconn.CommandTag, error)
	QueryRow(ctx context.Context, sql string, args ...any) pgx.Row
}

func providerForTask(taskType string) string {
	switch taskType {
	case "summary":
		return "groq"
	default:
		return "huggingface"
	}
}

func normalizeUUID(raw string, fallbackSeed string) uuid.UUID {
	raw = strings.TrimSpace(raw)
	if raw != "" {
		if parsed, err := uuid.Parse(raw); err == nil {
			return parsed
		}
		return uuid.NewSHA1(uuid.NameSpaceURL, []byte(raw))
	}

	if strings.TrimSpace(fallbackSeed) == "" {
		return uuid.New()
	}

	return uuid.NewSHA1(uuid.NameSpaceURL, []byte(fallbackSeed))
}

func sanitizeLabel(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "unknown"
	}

	s = strings.ReplaceAll(s, " ", "-")
	if len(s) > 48 {
		return s[:48]
	}

	return s
}

func batchTimestamps(batch *Batch, i int) (*time.Time, *time.Time) {
	if i < 0 || i >= batch.Size {
		return nil, nil
	}

	if batch.startedAtUnixNano[i] == 0 {
		return nil, nil
	}

	start := time.Unix(0, batch.startedAtUnixNano[i]).UTC()
	end := start.Add(time.Duration(batch.ProcessingMs[i]) * time.Millisecond)

	return &start, &end
}

type BatchPersistenceStats struct {
	RunsTotal   int
	RunsDone    int
	RunsFailed  int
	ResultsTotal int
}

func (s *Store) VerifyBatchPersistence(ctx context.Context, batch *Batch) (BatchPersistenceStats, error) {
	var stats BatchPersistenceStats

	if batch == nil || len(batch.AnalysisRunIDs) == 0 {
		return stats, nil
	}

	runIDs := append([]string(nil), batch.AnalysisRunIDs...)

	err := s.pool.QueryRow(ctx, `
		SELECT
			COUNT(*)::int AS runs_total,
			COUNT(*) FILTER (WHERE run_status = 'done')::int AS runs_done,
			COUNT(*) FILTER (WHERE run_status = 'failed')::int AS runs_failed
		FROM analysis_run
		WHERE id = ANY($1)
	`, runIDs).Scan(
		&stats.RunsTotal,
		&stats.RunsDone,
		&stats.RunsFailed,
	)
	if err != nil {
		return stats, fmt.Errorf("verify analysis_run stats: %w", err)
	}

	err = s.pool.QueryRow(ctx, `
		SELECT COUNT(*)::int
		FROM analysis_result
		WHERE analysis_run_id = ANY($1)
	`, runIDs).Scan(&stats.ResultsTotal)
	if err != nil {
		return stats, fmt.Errorf("verify analysis_result stats: %w", err)
	}

	return stats, nil
}
