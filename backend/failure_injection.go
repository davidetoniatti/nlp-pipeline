package main

import (
	"fmt"
	"os"
	"strings"
)

type FailureInjectionConfig struct {
	Enabled      bool
	DocIDs       map[string]struct{}
	TextContains string
}

func LoadFailureInjectionConfig() FailureInjectionConfig {
	docIDs := parseCSVSet(os.Getenv("FAIL_DOC_IDS"))
	textContains := strings.TrimSpace(os.Getenv("FAIL_TEXT_CONTAINS"))

	return FailureInjectionConfig{
		Enabled:      len(docIDs) > 0 || textContains != "",
		DocIDs:       docIDs,
		TextContains: textContains,
	}
}

func parseCSVSet(raw string) map[string]struct{} {
	out := make(map[string]struct{})
	for _, part := range strings.Split(raw, ",") {
		v := strings.TrimSpace(part)
		if v == "" {
			continue
		}
		out[v] = struct{}{}
	}
	return out
}

func (c FailureInjectionConfig) Match(docID, text string) (string, bool) {
	if len(c.DocIDs) > 0 {
		if _, ok := c.DocIDs[docID]; ok {
			return "doc_id matched FAIL_DOC_IDS", true
		}
	}

	if c.TextContains != "" && strings.Contains(strings.ToLower(text), strings.ToLower(c.TextContains)) {
		return fmt.Sprintf("text matched FAIL_TEXT_CONTAINS=%q", c.TextContains), true
	}

	return "", false
}

func applyFailureInjection(batch *Batch, cfg FailureInjectionConfig) int {
	if !cfg.Enabled {
		return 0
	}

	injected := 0
	for i := 0; i < batch.Size; i++ {
		reason, ok := cfg.Match(batch.IDs[i], batch.Texts[i])
		if !ok {
			continue
		}

		batch.ForceFail(i, fmt.Errorf("injected failure: %s", reason))
		injected++
	}

	return injected
}

func countBatchStatus(batch *Batch, wanted DocumentStatus) int {
	count := 0
	for _, s := range batch.Statuses {
		if s == wanted {
			count++
		}
	}
	return count
}
