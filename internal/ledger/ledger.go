package ledger

import (
	"fmt"
	"time"
)

// Alert represents a monitoring alert.
type Alert struct {
	ID        int64
	Timestamp time.Time
	SessionID string
	Metric    string
	Score     float64
	Threshold float64
	Action    string
}

// Intervention represents a steering intervention.
type Intervention struct {
	ID        int64
	Timestamp time.Time
	SessionID string
	Method    string
	Layer     int
	Result    string
}

// AlertFilter holds query parameters for alert filtering.
type AlertFilter struct {
	Metric    string
	Action    string
	SessionID string
	StartTime time.Time
	EndTime   time.Time
	Limit     int
	Offset    int
}

// Backend defines the interface for activity ledger storage.
type Backend interface {
	Init() error
	RecordAlert(alert Alert) error
	RecordIntervention(intervention Intervention) error
	QueryAlerts(filter AlertFilter) ([]Alert, error)
	QueryInterventions(sessionID string) ([]Intervention, error)
	CountAlerts(filter AlertFilter) (int, error)
	GetAlertStats() (map[string]int, error)
	DeleteOldAlerts(olderThan time.Duration) (int, error)
	ExportAlertsCSV(filter AlertFilter) (string, error)
	IngestBatch(alerts []Alert) error
	RetrieveBatch(sessionID string) ([]Alert, error)
	Close() error
}

// NewBackend creates a ledger backend.
func NewBackend(driver string, dsn string) (Backend, error) {
	switch driver {
	case "sqlite":
		return newSQLiteBackend(dsn)
	case "longbow":
		return newLongbowBackend(dsn)
	default:
		return nil, fmt.Errorf("unsupported driver: %s", driver)
	}
}
