package main

import (
	"sync"
	"time"
)

// DocumentStatus represents the processing status of a document.
type DocumentStatus uint8

const (
	StatusPending DocumentStatus = iota
	StatusProcessing
	StatusDone
	StatusFailed
)

// Batch contains a set of documents to be processed together.
//
// "Struct of arrays" design: instead of []Document (array of structs, each
// element with all its fields), we use parallel fields. This reduces
// allocation because:
//   - a single contiguous backing array per field
//   - better cache locality during access to a single field (e.g., only texts)
//   - no wasted padding/alignment between fields of different types
//
// All slices are pre-allocated with the batch capacity at creation time:
// no append = no dynamic reallocation.
//
// Concurrency contract:
//   - SetDocument must be called before any Worker starts (single-writer phase).
//   - SetResult and SetError may be called concurrently by Workers, each on a
//     DISTINCT index. The caller is responsible for index partitioning.
//   - FailedIndices and IsDone must be called only after all Workers have
//     completed (or under external synchronization).
type Batch struct {
	// mu protects Statuses, ProcessingErrors and batch-level timestamps during
	// concurrent Worker writes. RWMutex allows multiple concurrent readers
	// (FailedIndices, IsDone) while still serializing writes from concurrent
	// Workers calling SetResult / SetError / MarkProcessing.
	mu sync.RWMutex

	// Batch metadata.
	ID        string
	CreatedAt time.Time
	UpdatedAt time.Time // last modification time; useful for retry debugging
	Size      int       // number of documents in the batch
	PromptVersion string

	// Parallel fields: IDs[i], Texts[i], SourceIDs[i] refer to the
	// same document. Index access is O(1) and cache-friendly.
	IDs            []string // document UUIDs
	SourceIDs      []string // FK to source_metadata
	Hashes         []string // DOCUMENT.hash; optional but useful for dedup / traceability
	Texts          []string // original text (the heaviest field in memory)
	AnalysisRunIDs []string // analysis run UUIDs

	// Results: written by Workers concurrently.
	// We use fixed-size types where possible (float32 < float64).
	SentimentScores []float32      // [-1.0, 1.0]
	SentimentLabels []string       // "Positive", "Negative", "Neutral"
	Entities        [][]Entity     // slice of entities per document
	Summaries       []string       // only for negative documents
	ProcessingMs    []int32        // processing latency per document
	Statuses        []DocumentStatus
	ProcessingErrors []string // serialised error strings; avoids []error GC pressure

	// startedAtUnixNano tracks when each document entered StatusProcessing.
	// int64 is smaller and cheaper than storing []time.Time.
	startedAtUnixNano []int64

	// Traceability of the models used for this batch.
	ModelVersions BatchModelVersions
}

// Entity represents a named entity extracted from the text.
// It is a small value type: do not use pointers to avoid GC pressure.
type Entity struct {
	Text  string
	Label string  // PER, ORG, LOC, MISC
	Start int32   // character offset
	End   int32
	Score float32
}

// BatchModelVersions tracks which model version was used
// for each task in this batch. Fundamental for reproducibility.
type BatchModelVersions struct {
	SentimentModelID string
	NERModelID       string
	SummaryModelID   string
}

// NewBatch creates a Batch pre-allocated for size documents.
// Always pass the exact size: avoids internal slice reallocations.
func NewBatch(id string, size int) *Batch {
	now := time.Now().UTC()
	return &Batch{
		ID:        id,
		CreatedAt: now,
		UpdatedAt: now,
		Size:      size,

		// Pre-allocation with exact capacity: cap == len == size.
		// make([]T, size) allocates and zeroes in a single syscall.
		IDs:            make([]string, size),
		SourceIDs:      make([]string, size),
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

// SetDocument populates the i-th slot of the batch.
//
// Must be called before any Worker goroutine starts (single-writer phase).
// Concurrent calls from multiple goroutines are NOT safe; index partitioning
// alone does not protect the UpdatedAt field shared across all indices.
//
// This keeps backward compatibility with the previous signature and stores an
// empty hash when the caller does not have it yet.
func (b *Batch) SetDocument(i int, id, sourceID, text string) {
	b.SetDocumentWithHash(i, id, sourceID, text, "")
}

// SetDocumentWithHash populates the i-th slot including DOCUMENT.hash.
// Useful when the ingestion layer already computed a stable content hash.
func (b *Batch) SetDocumentWithHash(i int, id, sourceID, text, hash string) {
	b.IDs[i] = id
	b.SourceIDs[i] = sourceID
	b.Hashes[i] = hash
	b.Texts[i] = text
	b.AnalysisRunIDs[i] = ""

	// Ensure a clean slot in case the batch object is ever reused.
	b.SentimentScores[i] = 0
	b.SentimentLabels[i] = ""
	b.Entities[i] = nil
	b.Summaries[i] = ""
	b.ProcessingMs[i] = 0
	b.ProcessingErrors[i] = ""
	b.startedAtUnixNano[i] = 0

	b.Statuses[i] = StatusPending
}

// SetAnalysisRunID stores the ANALYSIS_RUN id for document i.
//
// Must be called before the Worker starts processing document i (single-writer
// phase), or under external synchronization if invoked concurrently.
func (b *Batch) SetAnalysisRunID(i int, runID string) {
	b.AnalysisRunIDs[i] = runID
}


// MarkProcessing marks document i as currently being processed.
//
// Workers should call this immediately before invoking the Python service.
// This makes StatusProcessing meaningful and allows latency measurement.
func (b *Batch) MarkProcessing(i int) {
	now := time.Now().UTC()

	b.mu.Lock()
	b.Statuses[i] = StatusProcessing
	b.startedAtUnixNano[i] = now.UnixNano()
	b.UpdatedAt = now
	b.mu.Unlock()
}

// SetResult writes the inference result for document i.
//
// Safe to call concurrently from multiple Workers provided each Worker owns a
// DISTINCT set of indices. Writing different indices of the same slice is safe
// in Go (no false sharing at the language level). UpdatedAt is updated under
// the write lock to avoid a data race on the shared timestamp field.
func (b *Batch) SetResult(i int, score float32, label string, entities []Entity, summary string) {
	// Copy entities defensively: the caller may reuse or mutate the input slice.
	clonedEntities := cloneEntities(entities)

	// Write result fields first — visible to readers only after StatusDone is set.
	b.SentimentScores[i] = score
	b.SentimentLabels[i] = label
	b.Entities[i] = clonedEntities
	b.Summaries[i] = summary
	b.ProcessingErrors[i] = ""

	started := b.startedAtUnixNano[i]
	if started != 0 {
		b.ProcessingMs[i] = elapsedMillisSince(started)
	}

	// Acquire the lock only for the shared fields that require serialization:
	// Statuses (memory-ordering guarantee for the status sentinel) and UpdatedAt.
	b.mu.Lock()
	b.Statuses[i] = StatusDone
	b.UpdatedAt = time.Now().UTC()
	b.mu.Unlock()
}

// SetError marks document i as failed. Thread-safe.
func (b *Batch) SetError(i int, err error) {
	now := time.Now().UTC()
	msg := "unknown error"
	if err != nil {
		msg = err.Error()
	}

	started := b.startedAtUnixNano[i]
	if started != 0 {
		b.ProcessingMs[i] = elapsedMillisSince(started)
	}

	b.mu.Lock()
	b.ProcessingErrors[i] = msg
	b.Statuses[i] = StatusFailed
	b.UpdatedAt = now
	b.mu.Unlock()
}

// FailedIndices returns the indices of failed documents.
// Used by the Orchestrator for retries.
//
// Must be called after all Workers have completed, or the caller must ensure
// no concurrent writes to Statuses are in flight.
func (b *Batch) FailedIndices() []int {
	b.mu.RLock()
	defer b.mu.RUnlock()

	// Capacity heuristic: expect at most 25% failures, minimum 1 to avoid
	// zero-cap allocation on small batches (Size < 4).
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

// IsDone reports whether all documents have reached a terminal state
// (StatusDone or StatusFailed). Safe to poll concurrently.
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

// TextsSlice returns only the texts of the specified indices,
// without allocating a new complete Batch structure.
// Useful for reprocessing only failed documents.
func (b *Batch) TextsSlice(indices []int) []string {
	out := make([]string, len(indices))
	for j, i := range indices {
		out[j] = b.Texts[i]
	}
	return out
}

// cloneEntities copies the slice header and its Entity values.
// Entity contains strings, which are immutable in Go, so copying the structs
// is enough here and avoids aliasing the original []Entity backing array.
func cloneEntities(in []Entity) []Entity {
	if len(in) == 0 {
		return nil
	}
	out := make([]Entity, len(in))
	copy(out, in)
	return out
}

// elapsedMillisSince converts a start timestamp to milliseconds.
// Saturates at MaxInt32 in the extremely unlikely event of a huge duration.
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

func (b *Batch) ForceFail(i int, err error) {
	b.SentimentScores[i] = 0
	b.SentimentLabels[i] = ""
	b.Entities[i] = nil
	b.Summaries[i] = ""
	b.SetError(i, err)
}
