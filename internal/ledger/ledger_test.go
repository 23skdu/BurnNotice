package ledger

import (
	"os"
	"strings"
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

func TestSQLite_Batch(t *testing.T) {
	tmp := t.TempDir() + "/sqlite_batch_test.db"
	defer os.Remove(tmp)

	backend, _ := NewBackend("sqlite", tmp)
	defer backend.Close()
	backend.Init()

	now := time.Now()
	batch := []Alert{
		{Timestamp: now, SessionID: "sess-batch-sqlite", Metric: "deception_score", Score: 0.85, Threshold: 0.7, Action: "steer"},
		{Timestamp: now, SessionID: "sess-batch-sqlite", Metric: "conflict_score", Score: 0.65, Threshold: 0.8, Action: "log"},
	}

	if err := backend.IngestBatch(batch); err != nil {
		t.Fatalf("SQLite IngestBatch failed: %v", err)
	}

	retrieved, err := backend.RetrieveBatch("sess-batch-sqlite")
	if err != nil {
		t.Fatalf("SQLite RetrieveBatch failed: %v", err)
	}
	if len(retrieved) != 2 {
		t.Errorf("retrieved len = %d, want 2", len(retrieved))
	}
}

func TestLongbow_InterventionsAndCSV(t *testing.T) {
	backend, err := NewBackend("longbow", "localhost:9090")
	if err != nil {
		t.Fatalf("failed to create longbow backend: %v", err)
	}
	defer backend.Close()
	backend.Init()

	now := time.Now()
	intervention := Intervention{
		Timestamp: now,
		SessionID: "sess-longbow-interventions",
		Method:    "ActAdd",
		Layer:     12,
		Result:    "success",
	}

	if err := backend.RecordIntervention(intervention); err != nil {
		t.Fatalf("Longbow RecordIntervention failed: %v", err)
	}

	retrieved, err := backend.QueryInterventions("sess-longbow-interventions")
	if err != nil {
		t.Fatalf("Longbow QueryInterventions failed: %v", err)
	}
	if len(retrieved) != 1 {
		t.Fatalf("retrieved interventions len = %d, want 1", len(retrieved))
	}
	if retrieved[0].Layer != 12 || retrieved[0].Method != "ActAdd" {
		t.Errorf("unexpected intervention retrieved: %+v", retrieved[0])
	}

	// Test ExportAlertsCSV on Longbow backend
	backend.RecordAlert(Alert{
		Timestamp: now,
		SessionID: "sess-longbow-interventions",
		Metric:    "deception_score",
		Score:     0.95,
		Threshold: 0.7,
		Action:    "terminate",
	})

	csv, err := backend.ExportAlertsCSV(AlertFilter{SessionID: "sess-longbow-interventions"})
	if err != nil {
		t.Fatalf("ExportAlertsCSV failed: %v", err)
	}
	if !strings.Contains(csv, "deception_score") || !strings.Contains(csv, "terminate") {
		t.Errorf("expected CSV to contain deception_score and terminate, got: %s", csv)
	}
}

func TestSQLite_ClosedDBErrors(t *testing.T) {
	tmp := t.TempDir() + "/closed_test.db"
	defer os.Remove(tmp)

	backend, err := NewBackend("sqlite", tmp)
	if err != nil {
		t.Fatal(err)
	}
	if err := backend.Init(); err != nil {
		t.Fatal(err)
	}

	// Close the connection
	if err := backend.Close(); err != nil {
		t.Fatal(err)
	}

	// Operations should return error
	alert := Alert{Timestamp: time.Now(), SessionID: "s", Metric: "m"}
	if err := backend.RecordAlert(alert); err == nil {
		t.Error("expected error RecordAlert on closed DB")
	}

	intervention := Intervention{Timestamp: time.Now(), SessionID: "s", Method: "m"}
	if err := backend.RecordIntervention(intervention); err == nil {
		t.Error("expected error RecordIntervention on closed DB")
	}

	if _, err := backend.QueryAlerts(AlertFilter{}); err == nil {
		t.Error("expected error QueryAlerts on closed DB")
	}

	if _, err := backend.QueryInterventions("s"); err == nil {
		t.Error("expected error QueryInterventions on closed DB")
	}

	if _, err := backend.CountAlerts(AlertFilter{}); err == nil {
		t.Error("expected error CountAlerts on closed DB")
	}

	if _, err := backend.GetAlertStats(); err == nil {
		t.Error("expected error GetAlertStats on closed DB")
	}

	if _, err := backend.DeleteOldAlerts(1 * time.Hour); err == nil {
		t.Error("expected error DeleteOldAlerts on closed DB")
	}

	if err := backend.IngestBatch([]Alert{alert}); err == nil {
		t.Error("expected error IngestBatch on closed DB")
	}
}

func TestLongbow_PaginationAndJSON(t *testing.T) {
	backend, err := NewBackend("longbow", "") // address fallback to default
	if err != nil {
		t.Fatal(err)
	}
	defer backend.Close()

	if err := backend.Init(); err != nil {
		t.Fatal(err)
	}

	now := time.Now()
	// Insert 2 alerts
	a1 := Alert{Timestamp: now, SessionID: "sess-p", Metric: "m1", Score: 0.5}
	a2 := Alert{Timestamp: now.Add(time.Minute), SessionID: "sess-p", Metric: "m2", Score: 0.6}
	_ = backend.RecordAlert(a1)
	_ = backend.RecordAlert(a2)

	// Filter.Offset >= len(results) (which is 2)
	results, err := backend.QueryAlerts(AlertFilter{SessionID: "sess-p", Offset: 5})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 0 {
		t.Errorf("expected empty results for offset overflow, got %d", len(results))
	}

	// Test Serialize/Deserialize
	data, err := SerializeAlert(a1)
	if err != nil {
		t.Fatal(err)
	}

	deserialized, err := DeserializeAlert(data)
	if err != nil {
		t.Fatal(err)
	}
	if deserialized.Metric != "m1" {
		t.Errorf("deserialized metric = %s, want m1", deserialized.Metric)
	}

	// Test DeserializeAlert with malformed JSON
	_, err = DeserializeAlert([]byte("invalid json"))
	if err == nil {
		t.Error("expected error deserializing malformed JSON, got nil")
	}
}

func TestSQLite_DefaultDSN(t *testing.T) {
	defer os.Remove("ledger.db")
	defer os.Remove("ledger.db-journal")
	defer os.Remove("ledger.db-shm")
	defer os.Remove("ledger.db-wal")

	backend, err := NewBackend("sqlite", "")
	if err != nil {
		t.Fatal(err)
	}
	defer backend.Close()
	if err := backend.Init(); err != nil {
		t.Fatal(err)
	}
}

func TestSQLite_FiltersAndSorting(t *testing.T) {
	tmp := t.TempDir() + "/filters_test.db"
	defer os.Remove(tmp)

	backend, err := NewBackend("sqlite", tmp)
	if err != nil {
		t.Fatal(err)
	}
	defer backend.Close()
	_ = backend.Init()

	now := time.Now()
	// Insert multiple alerts
	a1 := Alert{Timestamp: now.Add(-5 * time.Minute), SessionID: "sess-1", Metric: "m1", Score: 0.5, Threshold: 0.4, Action: "act-1"}
	a2 := Alert{Timestamp: now.Add(-10 * time.Minute), SessionID: "sess-2", Metric: "m2", Score: 0.6, Threshold: 0.4, Action: "act-2"}
	_ = backend.RecordAlert(a1)
	_ = backend.RecordAlert(a2)

	// Filter by Action
	res, err := backend.QueryAlerts(AlertFilter{Action: "act-1"})
	if err != nil || len(res) != 1 || res[0].Action != "act-1" {
		t.Errorf("QueryAlerts Action filter failed: %v", res)
	}

	// Filter by StartTime/EndTime
	res, err = backend.QueryAlerts(AlertFilter{StartTime: now.Add(-7 * time.Minute)})
	if err != nil || len(res) != 1 || res[0].SessionID != "sess-1" {
		t.Errorf("QueryAlerts StartTime filter failed: %v", res)
	}

	res, err = backend.QueryAlerts(AlertFilter{EndTime: now.Add(-7 * time.Minute)})
	if err != nil || len(res) != 1 || res[0].SessionID != "sess-2" {
		t.Errorf("QueryAlerts EndTime filter failed: %v", res)
	}

	// CountAlerts with filters
	c, err := backend.CountAlerts(AlertFilter{Action: "act-1"})
	if err != nil || c != 1 {
		t.Errorf("CountAlerts Action filter failed: %d, %v", c, err)
	}

	c, err = backend.CountAlerts(AlertFilter{SessionID: "sess-2"})
	if err != nil || c != 1 {
		t.Errorf("CountAlerts SessionID filter failed: %d, %v", c, err)
	}
}

func TestLongbow_FiltersAndSorting(t *testing.T) {
	backend, err := NewBackend("longbow", "localhost:9091")
	if err != nil {
		t.Fatal(err)
	}
	defer backend.Close()
	_ = backend.Init()

	now := time.Now()
	a1 := Alert{Timestamp: now.Add(-10 * time.Minute), SessionID: "sess-1", Metric: "m1", Score: 0.5, Threshold: 0.4, Action: "act-1"}
	a2 := Alert{Timestamp: now.Add(-5 * time.Minute), SessionID: "sess-2", Metric: "m2", Score: 0.6, Threshold: 0.4, Action: "act-2"}
	_ = backend.RecordAlert(a1)
	_ = backend.RecordAlert(a2)

	// Filter by Action
	res, err := backend.QueryAlerts(AlertFilter{Action: "act-1"})
	if err != nil || len(res) != 1 || res[0].Action != "act-1" {
		t.Errorf("Longbow QueryAlerts Action filter failed: %v", res)
	}

	// Mismatched SessionID filter to exercise negative check in QueryAlerts
	res, err = backend.QueryAlerts(AlertFilter{SessionID: "nonexistent"})
	if err != nil || len(res) != 0 {
		t.Errorf("expected 0 results, got %v", res)
	}

	// Filter by Metric
	res, err = backend.QueryAlerts(AlertFilter{Metric: "m2"})
	if err != nil || len(res) != 1 || res[0].Metric != "m2" {
		t.Errorf("Longbow QueryAlerts Metric filter failed: %v", res)
	}

	// Filter by StartTime/EndTime
	res, err = backend.QueryAlerts(AlertFilter{StartTime: now.Add(-7 * time.Minute)})
	if err != nil || len(res) != 1 || res[0].SessionID != "sess-2" {
		t.Errorf("Longbow QueryAlerts StartTime filter failed: %v", res)
	}

	res, err = backend.QueryAlerts(AlertFilter{EndTime: now.Add(-7 * time.Minute)})
	if err != nil || len(res) != 1 || res[0].SessionID != "sess-1" {
		t.Errorf("Longbow QueryAlerts EndTime filter failed: %v", res)
	}

	// Pagination offset / limit where limit < len(results)
	res, err = backend.QueryAlerts(AlertFilter{Limit: 1, Offset: 0})
	if err != nil || len(res) != 1 || res[0].SessionID != "sess-2" {
		t.Errorf("Longbow Pagination Limit/Offset failed: %v", res)
	}

	// Test QueryInterventions filters and sorting
	// Place them in ascending order of timestamps so the bubble sort has to swap them
	i1 := Intervention{Timestamp: now.Add(-10 * time.Minute), SessionID: "sess-1", Method: "ActAdd", Layer: 2}
	i2 := Intervention{Timestamp: now.Add(-5 * time.Minute), SessionID: "sess-1", Method: "ActAdd", Layer: 1}
	i3 := Intervention{Timestamp: now.Add(-20 * time.Minute), SessionID: "sess-2", Method: "ActAdd", Layer: 3}
	_ = backend.RecordIntervention(i1)
	_ = backend.RecordIntervention(i2)
	_ = backend.RecordIntervention(i3)

	interventions, err := backend.QueryInterventions("sess-1")
	if err != nil || len(interventions) != 2 {
		t.Errorf("Longbow QueryInterventions session filter failed: %v", interventions)
	}
	if interventions[0].Layer != 1 || interventions[1].Layer != 2 {
		t.Errorf("Longbow QueryInterventions sorting failed: %v", interventions)
	}

	// QueryInterventions with mismatched session ID to hit the continue block
	interventions, err = backend.QueryInterventions("nonexistent")
	if err != nil || len(interventions) != 0 {
		t.Errorf("expected 0 results, got %v", interventions)
	}

	// Test Longbow pagination offset where 0 < Offset < len(results)
	resOffset, err := backend.QueryAlerts(AlertFilter{Offset: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(resOffset) != 1 {
		t.Errorf("expected 1 result with offset 1, got %d", len(resOffset))
	}
}

func TestSQLite_ScanErrors(t *testing.T) {
	tmp := t.TempDir() + "/scan_error.db"
	defer os.Remove(tmp)

	backend, err := NewBackend("sqlite", tmp)
	if err != nil {
		t.Fatal(err)
	}
	defer backend.Close()

	if err := backend.Init(); err != nil {
		t.Fatal(err)
	}

	// Cast the backend to internal sqliteBackend
	sqliteB, ok := backend.(*sqliteBackend)
	if !ok {
		t.Fatal("expected sqliteBackend")
	}

	// Drop alerts and create a schema where score is a text and insert a bad string
	if _, err := sqliteB.db.Exec("DROP TABLE alerts"); err != nil {
		t.Fatal(err)
	}
	if _, err := sqliteB.db.Exec(`
		CREATE TABLE alerts (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			timestamp TEXT NOT NULL,
			session_id TEXT NOT NULL,
			metric TEXT NOT NULL,
			score TEXT NOT NULL,
			threshold REAL NOT NULL,
			action TEXT NOT NULL
		)
	`); err != nil {
		t.Fatal(err)
	}

	if _, err := sqliteB.db.Exec("INSERT INTO alerts (timestamp, session_id, metric, score, threshold, action) VALUES (?, ?, ?, ?, ?, ?)",
		time.Now().Format(time.RFC3339), "s-1", "m-1", "not-a-float", 0.5, "act-1"); err != nil {
		t.Fatal(err)
	}

	// Now querying alerts should trigger rows.Scan error
	if _, err := backend.QueryAlerts(AlertFilter{}); err == nil {
		t.Error("expected error QueryAlerts on malformed score value")
	}

	// Drop interventions and create a schema where layer is a text and insert a bad string
	if _, err := sqliteB.db.Exec("DROP TABLE interventions"); err != nil {
		t.Fatal(err)
	}
	if _, err := sqliteB.db.Exec(`
		CREATE TABLE interventions (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			timestamp TEXT NOT NULL,
			session_id TEXT NOT NULL,
			method TEXT NOT NULL,
			layer TEXT NOT NULL,
			result TEXT NOT NULL
		)
	`); err != nil {
		t.Fatal(err)
	}

	if _, err := sqliteB.db.Exec("INSERT INTO interventions (timestamp, session_id, method, layer, result) VALUES (?, ?, ?, ?, ?)",
		time.Now().Format(time.RFC3339), "s-1", "method-1", "not-an-int", "result-1"); err != nil {
		t.Fatal(err)
	}

	// Now querying interventions should trigger rows.Scan error
	if _, err := backend.QueryInterventions("s-1"); err == nil {
		t.Error("expected error QueryInterventions on malformed layer value")
	}
}
