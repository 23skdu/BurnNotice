package templates

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

func TestTemplates_LayoutAndDashboard(t *testing.T) {
	ctx := context.Background()

	// 1. Test Dashboard component (which embeds Layout)
	var bufDashboard bytes.Buffer
	err := Dashboard().Render(ctx, &bufDashboard)
	if err != nil {
		t.Fatalf("failed to render Dashboard: %v", err)
	}

	htmlDashboard := bufDashboard.String()
	if !strings.Contains(htmlDashboard, "System Overview") {
		t.Error("rendered Dashboard missing expected content: 'System Overview'")
	}
	if !strings.Contains(htmlDashboard, "BurnNotice") {
		t.Error("rendered Dashboard missing layout content: 'BurnNotice'")
	}

	// 2. Test Layout component directly
	var bufLayout bytes.Buffer
	err = Layout("CustomTitle").Render(ctx, &bufLayout)
	if err != nil {
		t.Fatalf("failed to render Layout: %v", err)
	}

	htmlLayout := bufLayout.String()
	if !strings.Contains(htmlLayout, "CustomTitle - BurnNotice") {
		t.Error("rendered Layout missing expected title: 'CustomTitle - BurnNotice'")
	}
	if !strings.Contains(htmlLayout, "https://unpkg.com/htmx.org") {
		t.Error("rendered Layout missing htmx script inclusion")
	}
}
