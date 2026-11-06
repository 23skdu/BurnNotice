package ledger

import (
	"database/sql"
	"fmt"
	"strings"
	"time"

	_ "modernc.org/sqlite"
)

type sqliteBackend struct {
	db *sql.DB
}

func newSQLiteBackend(dsn string) (*sqliteBackend, error) {
	if dsn == "" {
		dsn = "file:ledger.db?_journal_mode=WAL"
	}
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}
	return &sqliteBackend{db: db}, nil
}

func (b *sqliteBackend) Init() error {
	schema := `
	CREATE TABLE IF NOT EXISTS alerts (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		timestamp TEXT NOT NULL,
		session_id TEXT NOT NULL,
		metric TEXT NOT NULL,
		score REAL NOT NULL,
		threshold REAL NOT NULL,
		action TEXT NOT NULL
	);
	CREATE TABLE IF NOT EXISTS interventions (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		timestamp TEXT NOT NULL,
		session_id TEXT NOT NULL,
		method TEXT NOT NULL,
		layer INTEGER NOT NULL,
		result TEXT NOT NULL
	);
	CREATE INDEX IF NOT EXISTS idx_alerts_session ON alerts(session_id);
	CREATE INDEX IF NOT EXISTS idx_alerts_metric ON alerts(metric);
	CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
	CREATE INDEX IF NOT EXISTS idx_interventions_session ON interventions(session_id);
	`
	_, err := b.db.Exec(schema)
	return err
}

func (b *sqliteBackend) RecordAlert(alert Alert) error {
	_, err := b.db.Exec(
		"INSERT INTO alerts (timestamp, session_id, metric, score, threshold, action) VALUES (?, ?, ?, ?, ?, ?)",
		alert.Timestamp.Format(time.RFC3339),
		alert.SessionID,
		alert.Metric,
		alert.Score,
		alert.Threshold,
		alert.Action,
	)
	return err
}

func (b *sqliteBackend) RecordIntervention(intervention Intervention) error {
	_, err := b.db.Exec(
		"INSERT INTO interventions (timestamp, session_id, method, layer, result) VALUES (?, ?, ?, ?, ?)",
		intervention.Timestamp.Format(time.RFC3339),
		intervention.SessionID,
		intervention.Method,
		intervention.Layer,
		intervention.Result,
	)
	return err
}

func (b *sqliteBackend) QueryAlerts(filter AlertFilter) ([]Alert, error) {
	query := "SELECT id, timestamp, session_id, metric, score, threshold, action FROM alerts WHERE 1=1"
	args := []interface{}{}

	if filter.Metric != "" {
		query += " AND metric = ?"
		args = append(args, filter.Metric)
	}
	if filter.Action != "" {
		query += " AND action = ?"
		args = append(args, filter.Action)
	}
	if filter.SessionID != "" {
		query += " AND session_id = ?"
		args = append(args, filter.SessionID)
	}
	if !filter.StartTime.IsZero() {
		query += " AND timestamp >= ?"
		args = append(args, filter.StartTime.Format(time.RFC3339))
	}
	if !filter.EndTime.IsZero() {
		query += " AND timestamp <= ?"
		args = append(args, filter.EndTime.Format(time.RFC3339))
	}

	query += " ORDER BY timestamp DESC"

	if filter.Limit > 0 {
		query += " LIMIT ?"
		args = append(args, filter.Limit)
	}
	if filter.Offset > 0 {
		query += " OFFSET ?"
		args = append(args, filter.Offset)
	}

	rows, err := b.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	var alerts []Alert
	for rows.Next() {
		var a Alert
		var ts string
		if err := rows.Scan(&a.ID, &ts, &a.SessionID, &a.Metric, &a.Score, &a.Threshold, &a.Action); err != nil {
			return nil, err
		}
		a.Timestamp, _ = time.Parse(time.RFC3339, ts)
		alerts = append(alerts, a)
	}
	return alerts, rows.Err()
}

func (b *sqliteBackend) QueryInterventions(sessionID string) ([]Intervention, error) {
	rows, err := b.db.Query(
		"SELECT id, timestamp, session_id, method, layer, result FROM interventions WHERE session_id = ? ORDER BY timestamp DESC",
		sessionID,
	)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	var interventions []Intervention
	for rows.Next() {
		var i Intervention
		var ts string
		if err := rows.Scan(&i.ID, &ts, &i.SessionID, &i.Method, &i.Layer, &i.Result); err != nil {
			return nil, err
		}
		i.Timestamp, _ = time.Parse(time.RFC3339, ts)
		interventions = append(interventions, i)
	}
	return interventions, rows.Err()
}

func (b *sqliteBackend) Close() error {
	return b.db.Close()
}

// CountAlerts returns the total number of alerts matching a filter.
func (b *sqliteBackend) CountAlerts(filter AlertFilter) (int, error) {
	query := "SELECT COUNT(*) FROM alerts WHERE 1=1"
	args := []interface{}{}

	if filter.Metric != "" {
		query += " AND metric = ?"
		args = append(args, filter.Metric)
	}
	if filter.Action != "" {
		query += " AND action = ?"
		args = append(args, filter.Action)
	}
	if filter.SessionID != "" {
		query += " AND session_id = ?"
		args = append(args, filter.SessionID)
	}

	var count int
	err := b.db.QueryRow(query, args...).Scan(&count)
	return count, err
}

// GetAlertStats returns alert counts grouped by metric.
func (b *sqliteBackend) GetAlertStats() (map[string]int, error) {
	rows, err := b.db.Query("SELECT metric, COUNT(*) FROM alerts GROUP BY metric")
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	stats := make(map[string]int)
	for rows.Next() {
		var metric string
		var count int
		if err := rows.Scan(&metric, &count); err != nil {
			return nil, err
		}
		stats[metric] = count
	}
	return stats, rows.Err()
}

// DeleteOldAlerts deletes alerts older than the given duration.
func (b *sqliteBackend) DeleteOldAlerts(olderThan time.Duration) (int, error) {
	cutoff := time.Now().Add(-olderThan)
	result, err := b.db.Exec("DELETE FROM alerts WHERE timestamp < ?", cutoff.Format(time.RFC3339))
	if err != nil {
		return 0, err
	}
	rows, err := result.RowsAffected()
	if err != nil {
		return 0, err
	}
	return int(rows), nil
}

// ExportAlertsCSV exports alerts to CSV format.
func (b *sqliteBackend) ExportAlertsCSV(filter AlertFilter) (string, error) {
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

// IngestBatch inserts multiple alerts in a transaction.
func (b *sqliteBackend) IngestBatch(alerts []Alert) error {
	tx, err := b.db.Begin()
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()

	stmt, err := tx.Prepare("INSERT INTO alerts (timestamp, session_id, metric, score, threshold, action) VALUES (?, ?, ?, ?, ?, ?)")
	if err != nil {
		return err
	}
	defer func() { _ = stmt.Close() }()

	for _, a := range alerts {
		if _, err := stmt.Exec(a.Timestamp.Format(time.RFC3339), a.SessionID, a.Metric, a.Score, a.Threshold, a.Action); err != nil {
			return err
		}
	}

	return tx.Commit()
}

// RetrieveBatch retrieves all alerts for a session.
func (b *sqliteBackend) RetrieveBatch(sessionID string) ([]Alert, error) {
	return b.QueryAlerts(AlertFilter{SessionID: sessionID})
}
