package pipeline

import (
	"fmt"
	"os"
	"strings"
)

/*
ResilienceSimulatorConfig allows injecting controlled failures to test
error handling, retries, and persistence integrity.
*/
type ResilienceSimulatorConfig struct {
	Enabled      bool
	DocIDs       map[string]struct{}
	TextContains string
}

/*
LoadResilienceSimulatorConfig reads simulation triggers from environment variables.
Example: FAIL_DOC_IDS=doc-1,doc-2 or FAIL_TEXT_CONTAINS=error
*/
func LoadResilienceSimulatorConfig() ResilienceSimulatorConfig {
	docIDs := parseCSVSet(os.Getenv("FAIL_DOC_IDS"))
	textContains := strings.TrimSpace(os.Getenv("FAIL_TEXT_CONTAINS"))

	return ResilienceSimulatorConfig{
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

func (c ResilienceSimulatorConfig) Match(docID, text string) (string, bool) {
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

/*
ApplySimulation triggers failures on documents that match the config.
This should be called right before persistence to simulate late-stage failures.
*/
func ApplySimulation(batch *Batch, cfg ResilienceSimulatorConfig) int {
	if !cfg.Enabled {
		return 0
	}

	injected := 0
	for i := 0; i < batch.Size; i++ {
		reason, ok := cfg.Match(batch.IDs[i], batch.Texts[i])
		if !ok {
			continue
		}

		batch.ForceFail(i, fmt.Errorf("injected simulation failure: %s", reason))
		injected++
	}

	return injected
}
