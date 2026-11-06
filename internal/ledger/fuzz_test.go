package ledger

import (
	"math"
	"testing"
	"time"
	"unicode/utf8"
)

func FuzzSerializeDeserializeAlert(f *testing.F) {
	f.Add(int64(1), "sess-001", "deception_score", 0.85, 0.7, "steer")
	f.Add(int64(42), "sess-xyz", "conflict_score", 0.12, 0.5, "log")
	f.Add(int64(0), "", "", 0.0, 0.0, "")

	f.Fuzz(func(t *testing.T, id int64, sessionID, metric string, score, threshold float64, action string) {
		if math.IsNaN(score) || math.IsInf(score, 0) || math.IsNaN(threshold) || math.IsInf(threshold, 0) {
			return
		}
		if !utf8.ValidString(sessionID) || !utf8.ValidString(metric) || !utf8.ValidString(action) {
			return
		}

		orig := Alert{
			ID:        id,
			Timestamp: time.Unix(1700000000, 0).UTC(),
			SessionID: sessionID,
			Metric:    metric,
			Score:     score,
			Threshold: threshold,
			Action:    action,
		}

		data, err := SerializeAlert(orig)
		if err != nil {
			t.Fatalf("SerializeAlert error: %v", err)
		}

		parsed, err := DeserializeAlert(data)
		if err != nil {
			t.Fatalf("DeserializeAlert error: %v", err)
		}

		if parsed.ID != orig.ID || parsed.SessionID != orig.SessionID || parsed.Metric != orig.Metric || parsed.Action != orig.Action {
			t.Errorf("roundtrip mismatch: got %+v, want %+v", parsed, orig)
		}
	})
}

func FuzzLongbowBackendQuery(f *testing.F) {
	f.Add("sess-1", "deception", "block", 0.9, 10, 0)
	f.Add("sess-2", "truth", "log", 0.2, 1, 1)

	f.Fuzz(func(t *testing.T, sessionID, metric, action string, score float64, limit, offset int) {
		if math.IsNaN(score) || math.IsInf(score, 0) {
			return
		}
		if limit < -10 || limit > 1000 || offset < -10 || offset > 1000 {
			return
		}

		b, err := NewBackend("longbow", "localhost:9090")
		if err != nil {
			t.Fatalf("NewBackend error: %v", err)
		}
		defer func() { _ = b.Close() }()
		_ = b.Init()

		now := time.Unix(1700000000, 0).UTC()
		if err := b.RecordAlert(Alert{
			Timestamp: now,
			SessionID: sessionID,
			Metric:    metric,
			Score:     score,
			Threshold: 0.7,
			Action:    action,
		}); err != nil {
			t.Fatalf("RecordAlert failed: %v", err)
		}

		_, err = b.QueryAlerts(AlertFilter{
			SessionID: sessionID,
			Metric:    metric,
			Action:    action,
			Limit:     limit,
			Offset:    offset,
		})
		if err != nil {
			t.Fatalf("QueryAlerts failed: %v", err)
		}
	})
}
