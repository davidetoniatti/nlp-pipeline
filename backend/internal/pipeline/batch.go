package pipeline

import (
	"sync"
	"time"
)

/* DocumentStatus is the lifecycle state of a document inside a batch. */
type DocumentStatus uint8

const (
	StatusPending DocumentStatus = iota
	StatusProcessing
	StatusDone
	StatusFailed
)

/*
* Batch stores a group of documents processed together.
* The layout is "struct of arrays" instead of "array of structs".
* This keeps allocations predictable and makes per-field scans cheaper.
* Concurrency model:
* - SetDocument* methods are single-writer setup methods.
* - SetResult and SetError can run concurrently on different indices.
* - Readers like FailedIndices and IsDone use the mutex for consistency. */
type Batch struct {
	/* Protects shared state touched by concurrent workers. */
	mu sync.RWMutex

	/* Batch metadata. */
	ID            string
	CreatedAt     time.Time
	UpdatedAt     time.Time
	Size          int
	PromptVersion string
	PromptHash    string
	ServiceGitSHA string

	/* Parallel arrays. Same index, same document. */
	IDs            []string
	SourceIDs      []string
	Languages      []string
	Hashes         []string
	Texts          []string
	AnalysisRunIDs []string

	/* Per-document outputs. */
	SentimentScores  []float32
	SentimentLabels  []string
	Entities         [][]Entity
	Summaries        []string
	ProcessingMs     []int32
	Statuses         []DocumentStatus
	ProcessingErrors []string

	// When processing started for each document.
	// Unix nanoseconds are smaller and cheaper than []time.Time.

	startedAtUnixNano []int64

	/* Model traceability for the whole batch. */
	ModelVersions BatchModelVersions
}

/*
 * Entity is a named entity extracted from the text.
 * Keep it as a small value type. Pointers add no value here. */
type Entity struct {
	Text  string
	Label string
	Start int32
	End   int32
	Score float32
}

/*
BatchModelVersions records which models produced this batch.
This is small metadata, but it is important for traceability.
*/
type BatchModelVersions struct {
	Sentiment ModelMetadataDTO
	NER       ModelMetadataDTO
	Summary   ModelMetadataDTO
}

/* ModelMetadataDTO is the internal representation of model metadata. */
type ModelMetadataDTO struct {
	Name      string
	Version   string
	Revision  string
	Tokenizer string
	Provider  string
}

/*
NewBatch allocates a batch with all slices sized upfront.
The caller should pass the exact size so the batch never needs to grow.
*/
func NewBatch(id string, size int) *Batch {
	now := time.Now().UTC()

	return &Batch{
		ID:        id,
		CreatedAt: now,
		UpdatedAt: now,
		Size:      size,

		/* Fixed-size slices keep memory usage simple and predictable. */
		IDs:            make([]string, size),
		SourceIDs:      make([]string, size),
		Languages:      make([]string, size),
		Hashes:         make([]string, size),
		Texts:          make([]string, size),
		AnalysisRunIDs: make([]string, size),

		SentimentScores:  make([]float32, size),
		SentimentLabels:  make([]string, size),
		Entities:         make([][]Entity, size),
		Summaries:        make([]string, size),
		ProcessingMs:     make([]int32, size),
		Statuses:         make([]DocumentStatus, size),
		ProcessingErrors: make([]string, size),

		startedAtUnixNano: make([]int64, size),
	}
}

/*
SetDocumentWithLangAndHash fills one slot with all known input fields.
This must happen before workers start. It is setup code, not a concurrent path.
*/
func (b *Batch) SetDocumentWithLangAndHash(i int, id, sourceID, language, text, hash string) {
	b.IDs[i] = id
	b.SourceIDs[i] = sourceID
	b.Languages[i] = language
	b.Hashes[i] = hash
	b.Texts[i] = text
	b.AnalysisRunIDs[i] = ""

	/* Reset the slot so batch reuse does not leak stale state. */
	b.SentimentScores[i] = 0
	b.SentimentLabels[i] = ""
	b.Entities[i] = nil
	b.Summaries[i] = ""
	b.ProcessingMs[i] = 0
	b.ProcessingErrors[i] = ""
	b.startedAtUnixNano[i] = 0
	b.Statuses[i] = StatusPending
}

/*
SetAnalysisRunID assigns the logical run id for one document.
Retries reuse the same run id, so this field is set outside the hot path.
*/
func (b *Batch) SetAnalysisRunID(i int, runID string) {
	b.AnalysisRunIDs[i] = runID
}

/*
MarkProcessing moves a document into the processing state
and records when work actually started.
*/
func (b *Batch) MarkProcessing(i int) {
	now := time.Now().UTC()

	b.mu.Lock()
	b.Statuses[i] = StatusProcessing
	b.startedAtUnixNano[i] = now.UnixNano()
	b.UpdatedAt = now
	b.mu.Unlock()
}

/*
SetResult stores the successful output for one document.
Different workers may call this at the same time on different indices.
*/
func (b *Batch) SetResult(i int, score float32, label string, entities []Entity, summary string) {
	/* Clone entities so callers can safely reuse their input slice. */
	clonedEntities := cloneEntities(entities)

	/* Write payload first. Status is flipped only at the end. */
	b.SentimentScores[i] = score
	b.SentimentLabels[i] = label
	b.Entities[i] = clonedEntities
	b.Summaries[i] = summary
	b.ProcessingErrors[i] = ""

	if started := b.startedAtUnixNano[i]; started != 0 {
		b.ProcessingMs[i] = elapsedMillisSince(started)
	}

	/* Only shared state goes under the lock. */
	b.mu.Lock()
	b.Statuses[i] = StatusDone
	b.UpdatedAt = time.Now().UTC()
	b.mu.Unlock()
}

/* SetError marks one document as failed and stores a stable error string. */
func (b *Batch) SetError(i int, err error) {
	now := time.Now().UTC()
	msg := "unknown error"
	if err != nil {
		msg = err.Error()
	}

	if started := b.startedAtUnixNano[i]; started != 0 {
		b.ProcessingMs[i] = elapsedMillisSince(started)
	}

	b.mu.Lock()
	b.ProcessingErrors[i] = msg
	b.Statuses[i] = StatusFailed
	b.UpdatedAt = now
	b.mu.Unlock()
}

/*
FailedIndices returns the indices of documents that failed.
The small capacity hint avoids growing the output slice in common cases.
*/
func (b *Batch) FailedIndices() []int {
	b.mu.RLock()
	defer b.mu.RUnlock()

	capHint := b.Size / 4
	if capHint == 0 && b.Size > 0 {
		capHint = 1
	}

	failed := make([]int, 0, capHint)
	for i, s := range b.Statuses {
		if s == StatusFailed {
			failed = append(failed, i)
		}
	}
	return failed
}

/*
IsDone reports whether all documents reached a terminal state.
It is safe to poll while workers are still running.
*/
func (b *Batch) IsDone() bool {
	b.mu.RLock()
	defer b.mu.RUnlock()

	for _, s := range b.Statuses {
		if s == StatusPending || s == StatusProcessing {
			return false
		}
	}
	return true
}

/*
cloneEntities copies the entity slice so callers and batch state
do not share the same backing array.
*/
func cloneEntities(in []Entity) []Entity {
	if len(in) == 0 {
		return nil
	}

	out := make([]Entity, len(in))
	copy(out, in)
	return out
}

/*
elapsedMillisSince returns the elapsed time in milliseconds.
Clamp to int32 bounds even if absurdly large durations ever happen.
*/
func elapsedMillisSince(startUnixNano int64) int32 {
	elapsed := time.Since(time.Unix(0, startUnixNano)).Milliseconds()
	if elapsed < 0 {
		return 0
	}
	if elapsed > int64(^uint32(0)>>1) {
		return int32(^uint32(0) >> 1)
	}
	return int32(elapsed)
}

/*
ForceFail clears any successful-looking output and marks the slot as failed.
This is mainly useful for failure injection in tests and demos.
*/
func (b *Batch) ForceFail(i int, err error) {
	b.SentimentScores[i] = 0
	b.SentimentLabels[i] = ""
	b.Entities[i] = nil
	b.Summaries[i] = ""
	b.SetError(i, err)
}

/*
CountStatus returns the number of documents in the batch that have the given status.
This is useful for verification and reporting.
*/
func (b *Batch) CountStatus(wanted DocumentStatus) int {
	b.mu.RLock()
	defer b.mu.RUnlock()

	count := 0
	for _, s := range b.Statuses {
		if s == wanted {
			count++
		}
	}
	return count
}
