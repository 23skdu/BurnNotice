package registry

import (
	"testing"
	"time"
	"unicode/utf8"
)

func FuzzRegistryAddListFilter(f *testing.F) {
	f.Add("llama-3-8b", "llama", "llama")
	f.Add("qwen-2.5-0.5b", "qwen", "mistral")
	f.Add("gemma-2-2b", "gemma", "")

	f.Fuzz(func(t *testing.T, modelID, arch, filter string) {
		if modelID == "" || len(modelID) > 128 || len(filter) > 128 {
			return
		}
		if !utf8.ValidString(modelID) || !utf8.ValidString(arch) || !utf8.ValidString(filter) {
			return
		}

		r := NewRegistry(t.TempDir())
		r.Add(Metadata{
			ModelID:         modelID,
			Architecture:    arch,
			Date:            time.Unix(1700000000, 0).UTC(),
			DirectionCount:  8,
			ValidationScore: 0.9,
			Version:         1,
		})

		if r.Count() != 1 {
			t.Fatalf("Count() = %d, want 1", r.Count())
		}

		all := r.List("")
		if len(all) != 1 || all[0].ModelID != modelID {
			t.Fatalf("List(\"\") = %+v, want %s", all, modelID)
		}

		_ = r.List(filter)
	})
}
