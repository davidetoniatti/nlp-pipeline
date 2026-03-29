package pipeline

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

/* Store owns the database pool and persistence helpers. */
type Store struct {
	pool   *pgxpool.Pool
	logger *slog.Logger
}

/*
NewStore builds the Postgres store and verifies connectivity early.
Fail fast here so the rest of the system does not start half-alive.
*/
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

/* Close releases the underlying connection pool. */
func (s *Store) Close() {
	if s.pool != nil {
		s.pool.Close()
	}
}

/*
PersistBatch writes the whole batch in one transaction.
The batch is the unit of persistence: either everything lands,
or nothing does.
*/
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

	/*
		Resolve logical model names to real MODEL_VERSION rows first.
		Every analysis_run needs stable foreign keys.
	*/
	sentimentModelID, err := s.upsertModelVersion(ctx, tx, batch.ModelVersions.Sentiment, "sentiment")
	if err != nil {
		return fmt.Errorf("upsert sentiment model: %w", err)
	}

	nerModelID, err := s.upsertModelVersion(ctx, tx, batch.ModelVersions.NER, "ner")
	if err != nil {
		return fmt.Errorf("upsert ner model: %w", err)
	}

	summaryModelID, err := s.upsertModelVersion(ctx, tx, batch.ModelVersions.Summary, "summary")
	if err != nil {
		return fmt.Errorf("upsert summary model: %w", err)
	}

	for i := 0; i < batch.Size; i++ {
		/*
			Normalize all ids before touching the database.
			This lets the store accept either real UUIDs or stable logical ids.
		*/
		sourceUUID := normalizeUUID(batch.SourceIDs[i], "source:"+batch.SourceIDs[i])
		documentUUID := normalizeUUID(batch.IDs[i], "document:"+batch.IDs[i])
		analysisRunUUID := normalizeUUID(
			batch.AnalysisRunIDs[i],
			fmt.Sprintf("%s-run-%d", batch.ID, i),
		)

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
			analysisRunUUID,
			documentUUID,
			sentimentModelID,
			nerModelID,
			summaryModelID,
			startedAt,
			completedAt,
		); err != nil {
			return fmt.Errorf("upsert analysis_run %s: %w", analysisRunUUID, err)
		}

		/* analysis_result exists only for successful documents. */
		if batch.Statuses[i] == StatusDone {
			if err := s.upsertAnalysisResult(ctx, tx, batch, i, analysisRunUUID); err != nil {
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

/*
upsertSourceMetadata ensures the source row exists.
For the prototype the source payload is synthetic but still normalized
into a real relational table.
*/
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

/*
upsertDocument writes the input document row.
Optional fields are passed as NULL when the batch does not have them.
*/
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

	var languageValue any
	if strings.TrimSpace(batch.Languages[i]) != "" {
		languageValue = batch.Languages[i]
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
			language = EXCLUDED.language,
			status = EXCLUDED.status
	`,
		documentUUID,
		sourceUUID,
		batch.Texts[i],
		hashValue,
		languageValue,
		StatusString(batch.Statuses[i]),
		batch.CreatedAt,
	)

	return err
}

/*
upsertModelVersion ensures a model row exists and returns its id.
*/
func (s *Store) upsertModelVersion(ctx context.Context, tx pgxTx, m ModelMetadataDTO, taskType string) (uuid.UUID, error) {
	name := strings.TrimSpace(m.Name)
	if name == "" {
		name = "unknown"
	}

	version := strings.TrimSpace(m.Version)
	if version == "" {
		version = "unknown"
	}

	var revision any
	if strings.TrimSpace(m.Revision) != "" {
		revision = m.Revision
	}

	var tokenizer any
	if strings.TrimSpace(m.Tokenizer) != "" {
		tokenizer = m.Tokenizer
	}

	var id uuid.UUID
	err := tx.QueryRow(ctx, `
		INSERT INTO model_version (
			model_name,
			version,
			revision,
			tokenizer_name,
			task_type,
			provider,
			artifact_uri
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
		ON CONFLICT (model_name, version, revision, task_type, provider) DO UPDATE
		SET 
			tokenizer_name = EXCLUDED.tokenizer_name,
			artifact_uri = EXCLUDED.artifact_uri
		RETURNING id
	`,
		name,
		version,
		revision,
		tokenizer,
		taskType,
		m.Provider,
		name,
	).Scan(&id)

	if err != nil {
		return uuid.Nil, err
	}

	return id, nil
}

/*
upsertAnalysisRun writes the execution metadata for one document.
This row exists for both successful and failed runs.
*/
func (s *Store) upsertAnalysisRun(
	ctx context.Context,
	tx pgxTx,
	batch *Batch,
	i int,
	analysisRunUUID uuid.UUID,
	documentUUID uuid.UUID,
	sentimentModelID uuid.UUID,
	nerModelID uuid.UUID,
	summaryModelID uuid.UUID,
	startedAt, completedAt *time.Time,
) error {
	var promptVersion any
	if strings.TrimSpace(batch.PromptVersion) != "" {
		promptVersion = batch.PromptVersion
	}

	var promptHash any
	if strings.TrimSpace(batch.PromptHash) != "" {
		promptHash = batch.PromptHash
	}

	var serviceGitSHA any
	if strings.TrimSpace(batch.ServiceGitSHA) != "" {
		serviceGitSHA = batch.ServiceGitSHA
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
			prompt_hash,
			service_git_sha,
			started_at,
			completed_at,
			error_message
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
		ON CONFLICT (id) DO UPDATE
		SET
			document_id = EXCLUDED.document_id,
			sentiment_model_id = EXCLUDED.sentiment_model_id,
			ner_model_id = EXCLUDED.ner_model_id,
			summary_model_id = EXCLUDED.summary_model_id,
			run_status = EXCLUDED.run_status,
			trigger_type = EXCLUDED.trigger_type,
			prompt_version = EXCLUDED.prompt_version,
			prompt_hash = EXCLUDED.prompt_hash,
			service_git_sha = EXCLUDED.service_git_sha,
			started_at = EXCLUDED.started_at,
			completed_at = EXCLUDED.completed_at,
			error_message = EXCLUDED.error_message
	`,
		analysisRunUUID,
		documentUUID,
		sentimentModelID,
		nerModelID,
		summaryModelID,
		StatusString(batch.Statuses[i]),
		"batch_pipeline",
		promptVersion,
		promptHash,
		serviceGitSHA,
		startedAt,
		completedAt,
		errorMessage,
	)

	return err
}

/*
upsertAnalysisResult writes the successful output payload.
Entities are stored as JSONB because they are naturally nested data.
*/
func (s *Store) upsertAnalysisResult(
	ctx context.Context,
	tx pgxTx,
	batch *Batch,
	i int,
	analysisRunUUID uuid.UUID,
) error {
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
		analysisRunUUID,
		batch.SentimentScores[i],
		batch.SentimentLabels[i],
		string(entitiesJSON),
		summaryValue,
		batch.ProcessingMs[i],
	)

	return err
}

/*
pgxTx is the small transaction surface the store actually needs.
A narrow interface keeps helpers easy to test and decoupled from pgx.Tx.
*/
type pgxTx interface {
	Exec(ctx context.Context, sql string, arguments ...any) (pgconn.CommandTag, error)
	QueryRow(ctx context.Context, sql string, args ...any) pgx.Row
}

/*
normalizeUUID turns either a real UUID or a logical id into a stable UUID.
This is useful in a prototype where some inputs are symbolic ids
instead of true database-native UUIDs.
*/
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

/*
sanitizeLabel keeps synthetic labels readable and bounded.
This is not meant to be perfect slug logic, only good enough
for internal prototype data.
*/
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

/*
batchTimestamps reconstructs run timestamps from in-memory batch timing data.
If processing never started, there is nothing meaningful to persist.
*/
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

/* BatchPersistenceStats is a small verification summary for one batch. */
type BatchPersistenceStats struct {
	RunsTotal    int
	RunsDone     int
	RunsFailed   int
	ResultsTotal int
}

/*
VerifyBatchPersistence checks that the expected rows really landed in Postgres.
This is a cheap consistency check useful for the prototype and for demos.
*/
func (s *Store) VerifyBatchPersistence(ctx context.Context, batch *Batch) (BatchPersistenceStats, error) {
	var stats BatchPersistenceStats

	if batch == nil || len(batch.AnalysisRunIDs) == 0 {
		return stats, nil
	}

	runUUIDs := make([]uuid.UUID, 0, len(batch.AnalysisRunIDs))
	for i, raw := range batch.AnalysisRunIDs {
		runUUIDs = append(
			runUUIDs,
			normalizeUUID(raw, fmt.Sprintf("analysis-run:%s:%d", batch.ID, i)),
		)
	}

	err := s.pool.QueryRow(ctx, `
		SELECT
			COUNT(*)::int AS runs_total,
			COUNT(*) FILTER (WHERE run_status = 'done')::int AS runs_done,
			COUNT(*) FILTER (WHERE run_status = 'failed')::int AS runs_failed
		FROM analysis_run
		WHERE id = ANY($1)
	`, runUUIDs).Scan(
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
	`, runUUIDs).Scan(&stats.ResultsTotal)
	if err != nil {
		return stats, fmt.Errorf("verify analysis_result stats: %w", err)
	}

	return stats, nil
}

func StatusString(s DocumentStatus) string {
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
