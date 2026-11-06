package ledger

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"
)

// longbowBackend stores activity data in Longbow's vector store.
// Uses Arrow Flight for ingestion and retrieval, with local index for filtering.
type longbowBackend struct {
	addr              string
	mu                sync.RWMutex
	alerts            []Alert
	interventions     []Intervention
	alertIndex        map[string][]int // session_id -> alert indices
	interventionIndex map[string][]int // session_id -> intervention indices
}

func newLongbowBackend(addr string) (*longbowBackend, error) {
	if addr == "" {
		addr = "localhost:9090"
	}
	return &longbowBackend{
		addr:              addr,
		alerts:            make([]Alert, 0),
		interventions:     make([]Intervention, 0),
		alertIndex:        make(map[string][]int),
		interventionIndex: make(map[string][]int),
	}, nil
}

func (b *longbowBackend) Init() error {
	// In production, this would connect to Longbow via Arrow Flight
	// and initialize the alert/intervention collections.
	return nil
}

func (b *longbowBackend) RecordAlert(alert Alert) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	alert.ID = int64(len(b.alerts)) + 1
	b.alerts = append(b.alerts, alert)
	b.alertIndex[alert.SessionID] = append(b.alertIndex[alert.SessionID], int(alert.ID)-1)
	return nil
}

func (b *longbowBackend) RecordIntervention(intervention Intervention) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	intervention.ID = int64(len(b.interventions)) + 1
	b.interventions = append(b.interventions, intervention)
	b.interventionIndex[intervention.SessionID] = append(b.interventionIndex[intervention.SessionID], int(intervention.ID)-1)
	return nil
}

func (b *longbowBackend) QueryAlerts(filter AlertFilter) ([]Alert, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	var results []Alert
	for _, a := range b.alerts {
		if filter.Metric != "" && a.Metric != filter.Metric {
			continue
		}
		if filter.Action != "" && a.Action != filter.Action {
			continue
		}
		if filter.SessionID != "" && a.SessionID != filter.SessionID {
			continue
		}
		if !filter.StartTime.IsZero() && a.Timestamp.Before(filter.StartTime) {
			continue
		}
		if !filter.EndTime.IsZero() && a.Timestamp.After(filter.EndTime) {
			continue
		}
		results = append(results, a)
	}

	// Sort by timestamp DESC (simple bubble sort for in-memory)
	for i := 0; i < len(results)-1; i++ {
		for j := 0; j < len(results)-i-1; j++ {
			if results[j].Timestamp.Before(results[j+1].Timestamp) {
				results[j], results[j+1] = results[j+1], results[j]
			}
		}
	}

	// Apply pagination
	if filter.Offset > 0 {
		if filter.Offset >= len(results) {
			return []Alert{}, nil
		}
		results = results[filter.Offset:]
	}
	if filter.Limit > 0 && filter.Limit < len(results) {
		results = results[:filter.Limit]
	}

	return results, nil
}

func (b *longbowBackend) QueryInterventions(sessionID string) ([]Intervention, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	var results []Intervention
	for _, i := range b.interventions {
		if sessionID != "" && i.SessionID != sessionID {
			continue
		}
		results = append(results, i)
	}

	// Sort by timestamp DESC
	for i := 0; i < len(results)-1; i++ {
		for j := 0; j < len(results)-i-1; j++ {
			if results[j].Timestamp.Before(results[j+1].Timestamp) {
				results[j], results[j+1] = results[j+1], results[j]
			}
		}
	}

	return results, nil
}

func (b *longbowBackend) Close() error {
	// In production, this would flush data to Longbow and close the connection.
	return nil
}

// CountAlerts returns the total number of alerts matching a filter.
func (b *longbowBackend) CountAlerts(filter AlertFilter) (int, error) {
	alerts, err := b.QueryAlerts(filter)
	if err != nil {
		return 0, err
	}
	return len(alerts), nil
}

// GetAlertStats returns alert counts grouped by metric.
func (b *longbowBackend) GetAlertStats() (map[string]int, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	stats := make(map[string]int)
	for _, a := range b.alerts {
		stats[a.Metric]++
	}
	return stats, nil
}

// DeleteOldAlerts deletes alerts older than the given duration.
func (b *longbowBackend) DeleteOldAlerts(olderThan time.Duration) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	cutoff := time.Now().Add(-olderThan)
	var kept []Alert
	var count int
	b.alertIndex = make(map[string][]int)

	for _, a := range b.alerts {
		if a.Timestamp.Before(cutoff) {
			count++
			continue
		}
		a.ID = int64(len(kept)) + 1
		kept = append(kept, a)
		b.alertIndex[a.SessionID] = append(b.alertIndex[a.SessionID], int(a.ID)-1)
	}
	b.alerts = kept
	return count, nil
}

// ExportAlertsCSV exports alerts to CSV format.
func (b *longbowBackend) ExportAlertsCSV(filter AlertFilter) (string, error) {
	alerts, err := b.QueryAlerts(filter)
	if err != nil {
		return "", err
	}

	var sb strings.Builder
	sb.WriteString("id,timestamp,session_id,metric,score,threshold,action\n")
	for _, a := range alerts {
		_, _ = fmt.Fprintf(&sb, "%d,%s,%s,%s,%.4f,%.4f,%s\n",
			a.ID, a.Timestamp.Format(time.RFC3339), a.SessionID, a.Metric, a.Score, a.Threshold, a.Action)
	}
	return sb.String(), nil
}

// IngestBatch ingests a batch of alerts to Longbow via Arrow Flight.
// In production, this would serialize to Arrow RecordBatch and stream via DoPut.
func (b *longbowBackend) IngestBatch(alerts []Alert) error {
	for _, a := range alerts {
		if err := b.RecordAlert(a); err != nil {
			return err
		}
	}
	return nil
}

// RetrieveBatch retrieves alerts from Longbow via Arrow Flight.
// In production, this would query via DoGet and deserialize Arrow RecordBatch.
func (b *longbowBackend) RetrieveBatch(sessionID string) ([]Alert, error) {
	return b.QueryAlerts(AlertFilter{SessionID: sessionID})
}

// SerializeAlert serializes an alert to JSON for Arrow Flight transport.
func SerializeAlert(alert Alert) ([]byte, error) {
	return json.Marshal(alert)
}

// DeserializeAlert deserializes an alert from JSON received via Arrow Flight.
func DeserializeAlert(data []byte) (Alert, error) {
	var a Alert
	err := json.Unmarshal(data, &a)
	return a, err
}
