package ledger

import (
	"os"
	"testing"
	"time"
)

func TestNewBackend_SQLite(t *testing.T) {
	tmp := t.TempDir() + "/test.db"
	defer os.Remove(tmp)

	backend, err := NewBackend("sqlite", tmp)
	if err != nil {
		t.Fatalf("NewBackend error: %v", err)
	}
	defer backend.Close()

	if err := backend.Init(); err != nil {
		t.Fatalf("Init error: %v", err)
	}
}

func TestNewBackend_Longbow(t *testing.T) {
	backend, err := NewBackend("longbow", "localhost:9090")
	if err != nil {
		t.Fatalf("NewBackend error: %v", err)
	}
	defer backend.Close()

	if err := backend.Init(); err != nil {
		t.Fatalf("Init error: %v", err)
	}
}

func TestNewBackend_Unsupported(t *testing.T) {
	_, err := NewBackend("redis", "")
	if err == nil {
		t.Fatal("expected error for unsupported driver")
	}
}

func TestSQLite_RecordAlert(t *testing.T) {
	tmp := t.TempDir() + "/alert_test.db"
	defer os.Remove(tmp)

	backend, _ := NewBackend("sqlite", tmp)
	defer backend.Close()
	backend.Init()

	alert := Alert{
		Timestamp: time.Now(),
		SessionID: "sess-001",
		Metric:    "deception_score",
		Score:     0.85,
		Threshold: 0.7,
		Action:    "steer",
	}

	if err := backend.RecordAlert(alert); err != nil {
		t.Fatalf("RecordAlert error: %v", err)
	}
}

func TestSQLite_RecordIntervention(t *testing.T) {
	tmp := t.TempDir() + "/intervention_test.db"
	defer os.Remove(tmp)

	backend, _ := NewBackend("sqlite", tmp)
	defer backend.Close()
	backend.Init()

	intervention := Intervention{
		Timestamp: time.Now(),
		SessionID: "sess-001",
		Method:    "ActAdd",
		Layer:     5,
		Result:    "steered",
	}

	if err := backend.RecordIntervention(intervention); err != nil {
		t.Fatalf("RecordIntervention error: %v", err)
	}
}

func TestSQLite_QueryAlerts(t *testing.T) {
	tmp := t.TempDir() + "/query_test.db"
	defer os.Remove(tmp)

	backend, _ := NewBackend("sqlite", tmp)
	defer backend.Close()
	backend.Init()

	now := time.Now()
	alerts := []Alert{
		{Timestamp: now.Add(-2 * time.Hour), SessionID: "sess-001", Metric: "deception_score", Score: 0.8, Threshold: 0.7, Action: "steer"},
		{Timestamp: now.Add(-1 * time.Hour), SessionID: "sess-002", Metric: "conflict_score", Score: 0.6, Threshold: 0.8, Action: "log"},
		{Timestamp: now, SessionID: "sess-001", Metric: "deception_score", Score: 0.9, Threshold: 0.7, Action: "block"},
	}

	for _, a := range alerts {
		backend.RecordAlert(a)
	}

	// Query all
	all, err := backend.QueryAlerts(AlertFilter{})
	if err != nil {
		t.Fatalf("QueryAlerts error: %v", err)
	}
	if len(all) != 3 {
		t.Fatalf("len(all) = %d, want 3", len(all))
	}

	// Query by metric
	deception, err := backend.QueryAlerts(AlertFilter{Metric: "deception_score"})
	if err != nil {
		t.Fatalf("QueryAlerts error: %v", err)
	}
	if len(deception) != 2 {
		t.Fatalf("len(deception) = %d, want 2", len(deception))
	}

	// Query by session
	sess1, err := backend.QueryAlerts(AlertFilter{SessionID: "sess-001"})
	if err != nil {
		t.Fatalf("QueryAlerts error: %v", err)
	}
	if len(sess1) != 2 {
		t.Fatalf("len(sess1) = %d, want 2", len(sess1))
	}
}

func TestSQLite_QueryAlerts_Pagination(t *testing.T) {
	tmp := t.TempDir() + "/pagination_test.db"
	defer os.Remove(tmp)

	backend, _ := NewBackend("sqlite", tmp)
	defer backend.Close()
	backend.Init()

	now := time.Now()
	for i := 0; i < 10; i++ {
		backend.RecordAlert(Alert{
			Timestamp: now.Add(time.Duration(-i) * time.Minute),
			SessionID: "sess-001",
			Metric:    "deception_score",
			Score:     0.5,
			Threshold: 0.7,
			Action:    "log",
		})
	}

	// Limit
	limited, err := backend.QueryAlerts(AlertFilter{Limit: 3})
	if err != nil {
		t.Fatalf("QueryAlerts error: %v", err)
	}
	if len(limited) != 3 {
		t.Fatalf("len(limited) = %d, want 3", len(limited))
	}

	// Offset
	offset, err := backend.QueryAlerts(AlertFilter{Offset: 5, Limit: 3})
	if err != nil {
		t.Fatalf("QueryAlerts error: %v", err)
	}
	if len(offset) != 3 {
		t.Fatalf("len(offset) = %d, want 3", len(offset))
	}
}

func TestSQLite_QueryInterventions(t *testing.T) {
	tmp := t.TempDir() + "/query_intervention_test.db"
	defer os.Remove(tmp)

	backend, _ := NewBackend("sqlite", tmp)
	defer backend.Close()
	backend.Init()

	now := time.Now()
	backend.RecordIntervention(Intervention{Timestamp: now, SessionID: "sess-001", Method: "ActAdd", Layer: 5, Result: "steered"})
	backend.RecordIntervention(Intervention{Timestamp: now, SessionID: "sess-002", Method: "Angular", Layer: 3, Result: "steered"})

	interventions, err := backend.QueryInterventions("sess-001")
	if err != nil {
		t.Fatalf("QueryInterventions error: %v", err)
	}
	if len(interventions) != 1 {
		t.Fatalf("len(interventions) = %d, want 1", len(interventions))
	}
	if interventions[0].Method != "ActAdd" {
		t.Errorf("Method = %s, want ActAdd", interventions[0].Method)
	}
}

func TestSQLite_CountAlerts(t *testing.T) {
	tmp := t.TempDir() + "/count_test.db"
	defer os.Remove(tmp)

	backend, _ := NewBackend("sqlite", tmp)
	defer backend.Close()
	backend.Init()

	now := time.Now()
	backend.RecordAlert(Alert{Timestamp: now, SessionID: "sess-001", Metric: "deception_score", Score: 0.8, Threshold: 0.7, Action: "steer"})
	backend.RecordAlert(Alert{Timestamp: now, SessionID: "sess-002", Metric: "conflict_score", Score: 0.6, Threshold: 0.8, Action: "log"})

	count, err := backend.CountAlerts(AlertFilter{Metric: "deception_score"})
	if err != nil {
		t.Fatalf("CountAlerts error: %v", err)
	}
	if count != 1 {
		t.Errorf("count = %d, want 1", count)
	}
}

func TestSQLite_GetAlertStats(t *testing.T) {
	tmp := t.TempDir() + "/stats_test.db"
	defer os.Remove(tmp)

	backend, _ := NewBackend("sqlite", tmp)
	defer backend.Close()
	backend.Init()

	now := time.Now()
	backend.RecordAlert(Alert{Timestamp: now, SessionID: "sess-001", Metric: "deception_score", Score: 0.8, Threshold: 0.7, Action: "steer"})
	backend.RecordAlert(Alert{Timestamp: now, SessionID: "sess-002", Metric: "deception_score", Score: 0.9, Threshold: 0.7, Action: "block"})
	backend.RecordAlert(Alert{Timestamp: now, SessionID: "sess-003", Metric: "conflict_score", Score: 0.6, Threshold: 0.8, Action: "log"})

	stats, err := backend.GetAlertStats()
	if err != nil {
		t.Fatalf("GetAlertStats error: %v", err)
	}
	if stats["deception_score"] != 2 {
		t.Errorf("deception_score count = %d, want 2", stats["deception_score"])
	}
	if stats["conflict_score"] != 1 {
		t.Errorf("conflict_score count = %d, want 1", stats["conflict_score"])
	}
}

func TestSQLite_DeleteOldAlerts(t *testing.T) {
	tmp := t.TempDir() + "/delete_test.db"
	defer os.Remove(tmp)

	backend, _ := NewBackend("sqlite", tmp)
	defer backend.Close()
	backend.Init()

	now := time.Now()
	backend.RecordAlert(Alert{Timestamp: now.Add(-2 * time.Hour), SessionID: "sess-001", Metric: "deception_score", Score: 0.8, Threshold: 0.7, Action: "steer"})
	backend.RecordAlert(Alert{Timestamp: now, SessionID: "sess-002", Metric: "deception_score", Score: 0.9, Threshold: 0.7, Action: "block"})

	deleted, err := backend.DeleteOldAlerts(1 * time.Hour)
	if err != nil {
		t.Fatalf("DeleteOldAlerts error: %v", err)
	}
	if deleted != 1 {
		t.Errorf("deleted = %d, want 1", deleted)
	}

	remaining, _ := backend.QueryAlerts(AlertFilter{})
	if len(remaining) != 1 {
		t.Errorf("remaining = %d, want 1", len(remaining))
	}
}

func TestSQLite_ExportAlertsCSV(t *testing.T) {
	tmp := t.TempDir() + "/csv_test.db"
	defer os.Remove(tmp)

	backend, _ := NewBackend("sqlite", tmp)
	defer backend.Close()
	backend.Init()

	now := time.Now()
	backend.RecordAlert(Alert{Timestamp: now, SessionID: "sess-001", Metric: "deception_score", Score: 0.8, Threshold: 0.7, Action: "steer"})

	csv, err := backend.ExportAlertsCSV(AlertFilter{})
	if err != nil {
		t.Fatalf("ExportAlertsCSV error: %v", err)
	}
	if csv == "" {
		t.Fatal("CSV output is empty")
	}
}

func TestLongbowBackend_RecordAndQuery(t *testing.T) {
	backend, _ := NewBackend("longbow", "localhost:9090")
	defer backend.Close()
	backend.Init()

	now := time.Now()
	backend.RecordAlert(Alert{Timestamp: now, SessionID: "sess-001", Metric: "deception_score", Score: 0.8, Threshold: 0.7, Action: "steer"})
	backend.RecordAlert(Alert{Timestamp: now, SessionID: "sess-002", Metric: "conflict_score", Score: 0.6, Threshold: 0.8, Action: "log"})

	alerts, err := backend.QueryAlerts(AlertFilter{Metric: "deception_score"})
	if err != nil {
		t.Fatalf("QueryAlerts error: %v", err)
	}
	if len(alerts) != 1 {
		t.Fatalf("len(alerts) = %d, want 1", len(alerts))
	}
}

func TestLongbowBackend_IngestBatch(t *testing.T) {
	backend, _ := NewBackend("longbow", "localhost:9090")
	defer backend.Close()
	backend.Init()

	now := time.Now()
	batch := []Alert{
		{Timestamp: now, SessionID: "sess-001", Metric: "deception_score", Score: 0.8, Threshold: 0.7, Action: "steer"},
		{Timestamp: now, SessionID: "sess-002", Metric: "conflict_score", Score: 0.6, Threshold: 0.8, Action: "log"},
	}

	if err := backend.IngestBatch(batch); err != nil {
		t.Fatalf("IngestBatch error: %v", err)
	}

	count, _ := backend.CountAlerts(AlertFilter{})
	if count != 2 {
		t.Errorf("count = %d, want 2", count)
	}
}

func TestLongbowBackend_RetrieveBatch(t *testing.T) {
	backend, _ := NewBackend("longbow", "localhost:9090")
	defer backend.Close()
	backend.Init()

	now := time.Now()
	backend.RecordAlert(Alert{Timestamp: now, SessionID: "sess-001", Metric: "deception_score", Score: 0.8, Threshold: 0.7, Action: "steer"})

	retrieved, err := backend.RetrieveBatch("sess-001")
	if err != nil {
		t.Fatalf("RetrieveBatch error: %v", err)
	}
	if len(retrieved) != 1 {
		t.Fatalf("len(retrieved) = %d, want 1", len(retrieved))
	}
}

func TestSerializeDeserializeAlert(t *testing.T) {
	alert := Alert{
		ID:        1,
		Timestamp: time.Now(),
		SessionID: "sess-001",
		Metric:    "deception_score",
		Score:     0.85,
		Threshold: 0.7,
		Action:    "steer",
	}

	data, err := SerializeAlert(alert)
	if err != nil {
		t.Fatalf("SerializeAlert error: %v", err)
	}

	deserialized, err := DeserializeAlert(data)
	if err != nil {
		t.Fatalf("DeserializeAlert error: %v", err)
	}

	if deserialized.SessionID != alert.SessionID {
		t.Errorf("SessionID = %s, want %s", deserialized.SessionID, alert.SessionID)
	}
	if deserialized.Score != alert.Score {
		t.Errorf("Score = %f, want %f", deserialized.Score, alert.Score)
	}
}

func TestLongbowBackend_DeleteOldAlerts(t *testing.T) {
	backend, _ := NewBackend("longbow", "localhost:9090")
	defer backend.Close()
	backend.Init()

	now := time.Now()
	backend.RecordAlert(Alert{Timestamp: now.Add(-2 * time.Hour), SessionID: "sess-001", Metric: "deception_score", Score: 0.8, Threshold: 0.7, Action: "steer"})
	backend.RecordAlert(Alert{Timestamp: now, SessionID: "sess-002", Metric: "deception_score", Score: 0.9, Threshold: 0.7, Action: "block"})

	deleted, err := backend.DeleteOldAlerts(1 * time.Hour)
	if err != nil {
		t.Fatalf("DeleteOldAlerts error: %v", err)
	}
	if deleted != 1 {
		t.Errorf("deleted = %d, want 1", deleted)
	}
}

func TestLongbowBackend_GetAlertStats(t *testing.T) {
	backend, _ := NewBackend("longbow", "localhost:9090")
	defer backend.Close()
	backend.Init()

	now := time.Now()
	backend.RecordAlert(Alert{Timestamp: now, SessionID: "sess-001", Metric: "deception_score", Score: 0.8, Threshold: 0.7, Action: "steer"})
	backend.RecordAlert(Alert{Timestamp: now, SessionID: "sess-002", Metric: "deception_score", Score: 0.9, Threshold: 0.7, Action: "block"})
	backend.RecordAlert(Alert{Timestamp: now, SessionID: "sess-003", Metric: "conflict_score", Score: 0.6, Threshold: 0.8, Action: "log"})

	stats, err := backend.GetAlertStats()
	if err != nil {
		t.Fatalf("GetAlertStats error: %v", err)
	}
	if stats["deception_score"] != 2 {
		t.Errorf("deception_score count = %d, want 2", stats["deception_score"])
	}
}
