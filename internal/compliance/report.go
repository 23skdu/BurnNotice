package compliance

import (
	"bytes"
	"fmt"
	"time"

	"burnnotice/internal/ledger"
)

// Report holds a compliance report.
type Report struct {
	GeneratedAt     time.Time
	StartTime       time.Time
	EndTime         time.Time
	TotalAlerts     int
	AlertsByMetric  map[string]int
	AlertsByAction  map[string]int
	ComplianceScore float64
	EUAIAct         EUAIActReport
	NISTAIRMF       NISTAIRMFReport
}

// EUAIActReport holds EU AI Act compliance data.
type EUAIActReport struct {
	HighRiskAlerts     int
	LimitedRiskAlerts  int
	MinimalRiskAlerts  int
	ComplianceMetrics  map[string]float64
}

// NISTAIRMFReport holds NIST AI RMF compliance data.
type NISTAIRMFReport struct {
	Govern   int
	Map      int
	Measure  int
	Manage   int
}

// Reporter generates compliance reports.
type Reporter struct {
	ledgerBackend ledger.Backend
}

// NewReporter creates a new reporter.
func NewReporter(backend ledger.Backend) *Reporter {
	return &Reporter{ledgerBackend: backend}
}

// GenerateReport generates a compliance report for the given time range.
func (r *Reporter) GenerateReport(startTime, endTime time.Time) (*Report, error) {
	filter := ledger.AlertFilter{
		StartTime: startTime,
		EndTime:   endTime,
	}

	alerts, err := r.ledgerBackend.QueryAlerts(filter)
	if err != nil {
		return nil, fmt.Errorf("query alerts: %w", err)
	}

	report := &Report{
		GeneratedAt:    time.Now(),
		StartTime:      startTime,
		EndTime:        endTime,
		TotalAlerts:    len(alerts),
		AlertsByMetric: make(map[string]int),
		AlertsByAction: make(map[string]int),
	}

	for _, a := range alerts {
		report.AlertsByMetric[a.Metric]++
		report.AlertsByAction[a.Action]++
	}

	if report.TotalAlerts > 0 {
		blockedCount := report.AlertsByAction["block"] + report.AlertsByAction["terminate"]
		report.ComplianceScore = float64(blockedCount) / float64(report.TotalAlerts)
	} else {
		report.ComplianceScore = 1.0
	}

	report.EUAIAct = r.generateEUAIActReport(alerts)
	report.NISTAIRMF = r.generateNISTAIRMFReport(alerts)

	return report, nil
}

func (r *Reporter) generateEUAIActReport(alerts []ledger.Alert) EUAIActReport {
	eu := EUAIActReport{
		ComplianceMetrics: make(map[string]float64),
	}

	for _, a := range alerts {
		switch a.Metric {
		case "deception_score":
			if a.Score > 0.8 {
				eu.HighRiskAlerts++
			} else if a.Score > 0.5 {
				eu.LimitedRiskAlerts++
			} else {
				eu.MinimalRiskAlerts++
			}
		default:
			eu.MinimalRiskAlerts++
		}
	}

	if len(alerts) > 0 {
		eu.ComplianceMetrics["high_risk_ratio"] = float64(eu.HighRiskAlerts) / float64(len(alerts))
		eu.ComplianceMetrics["limited_risk_ratio"] = float64(eu.LimitedRiskAlerts) / float64(len(alerts))
	}

	return eu
}

func (r *Reporter) generateNISTAIRMFReport(alerts []ledger.Alert) NISTAIRMFReport {
	nist := NISTAIRMFReport{}

	for _, a := range alerts {
		switch a.Action {
		case "log":
			nist.Measure++
		case "steer":
			nist.Manage++
		case "block":
			nist.Govern++
		case "terminate":
			nist.Govern++
		default:
			nist.Map++
		}
	}

	return nist
}

// ExportReport exports a report to JSON string.
func (r *Reporter) ExportReport(report *Report) (string, error) {
	return fmt.Sprintf(`{"generated_at":"%s","total_alerts":%d,"compliance_score":%.4f}`,
		report.GeneratedAt.Format(time.RFC3339),
		report.TotalAlerts,
		report.ComplianceScore), nil
}

// ExportPDF generates a PDF report.
func (r *Reporter) ExportPDF(report *Report) ([]byte, error) {
	var buf bytes.Buffer

	buf.WriteString("%PDF-1.4\n")

	buf.WriteString("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")

	content := buildPDFContent(report)
	contentStream := fmt.Sprintf("3 0 obj\n<< /Length %d >>\nstream\n%s\nendstream\nendobj\n", len(content), content)
	buf.WriteString(contentStream)

	pageObj := "4 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 3 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
	buf.WriteString(pageObj)

	fontObj := "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
	buf.WriteString(fontObj)

	xrefOffset := buf.Len()
	buf.WriteString("xref\n0 6\n")
	buf.WriteString("0000000000 65535 f \n")
	buf.WriteString("0000000009 00000 n \n")

	contentStart := len("%PDF-1.4\n") + len("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
	buf.WriteString(fmt.Sprintf("%010d 00000 n \n", contentStart))

	pageStart := contentStart + len(contentStream)
	buf.WriteString(fmt.Sprintf("%010d 00000 n \n", pageStart))

	fontStart := pageStart + len(pageObj)
	buf.WriteString(fmt.Sprintf("%010d 00000 n \n", fontStart))

	trailer := fmt.Sprintf("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n", xrefOffset)
	buf.WriteString(trailer)

	return buf.Bytes(), nil
}

func buildPDFContent(report *Report) string {
	var content string

	content += "BT\n/F1 18 Tf 50 750 Td (BurnNotice Compliance Report) Tj ET\n"
	content += "BT\n/F1 12 Tf 50 720 Td (Generated: " + report.GeneratedAt.Format("2006-01-02 15:04:05") + ") Tj ET\n"
	content += "BT\n/F1 12 Tf 50 700 Td (Period: " + report.StartTime.Format("2006-01-02") + " to " + report.EndTime.Format("2006-01-02") + ") Tj ET\n"
	content += "BT\n/F1 14 Tf 50 670 Td (Summary) Tj ET\n"
	content += fmt.Sprintf("BT\n/F1 12 Tf 50 650 Td (Total Alerts: %d) Tj ET\n", report.TotalAlerts)
	content += fmt.Sprintf("BT\n/F1 12 Tf 50 630 Td (Compliance Score: %.2f) Tj ET\n", report.ComplianceScore)

	content += "BT\n/F1 14 Tf 50 590 Td (Alerts by Metric) Tj ET\n"
	y := 570
	for metric, count := range report.AlertsByMetric {
		content += fmt.Sprintf("BT\n/F1 10 Tf 50 %d Td (%s: %d) Tj ET\n", y, metric, count)
		y -= 15
	}

	content += "BT\n/F1 14 Tf 50 520 Td (EU AI Act Risk Classification) Tj ET\n"
	content += fmt.Sprintf("BT\n/F1 12 Tf 50 500 Td (High Risk: %d) Tj ET\n", report.EUAIAct.HighRiskAlerts)
	content += fmt.Sprintf("BT\n/F1 12 Tf 50 480 Td (Limited Risk: %d) Tj ET\n", report.EUAIAct.LimitedRiskAlerts)
	content += fmt.Sprintf("BT\n/F1 12 Tf 50 460 Td (Minimal Risk: %d) Tj ET\n", report.EUAIAct.MinimalRiskAlerts)

	content += "BT\n/F1 14 Tf 50 420 Td (NIST AI RMF) Tj ET\n"
	content += fmt.Sprintf("BT\n/F1 12 Tf 50 400 Td (Govern: %d) Tj ET\n", report.NISTAIRMF.Govern)
	content += fmt.Sprintf("BT\n/F1 12 Tf 50 380 Td (Map: %d) Tj ET\n", report.NISTAIRMF.Map)
	content += fmt.Sprintf("BT\n/F1 12 Tf 50 360 Td (Measure: %d) Tj ET\n", report.NISTAIRMF.Measure)
	content += fmt.Sprintf("BT\n/F1 12 Tf 50 340 Td (Manage: %d) Tj ET\n", report.NISTAIRMF.Manage)

	return content
}
