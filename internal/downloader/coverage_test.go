package downloader

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestDownload_ResumeRange(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/models/test-model/tree/main" {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`[{"path":"model.safetensors"}]`))
			return
		}
		// Simulate a resumed download: server honors Range and returns 206.
		if r.Header.Get("Range") != "" {
			w.WriteHeader(http.StatusPartialContent)
			_, _ = w.Write([]byte("tail"))
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("full content"))
	}))
	defer server.Close()

	outputDir := t.TempDir()
	// Pre-create a partial file to trigger the resume path.
	partial := filepath.Join(outputDir, "model.safetensors")
	if err := os.WriteFile(partial, []byte("head"), 0600); err != nil {
		t.Fatal(err)
	}

	d := NewHuggingFaceDownloader("").WithBaseURL(server.URL)
	if err := d.Download("test-model", outputDir, nil); err != nil {
		t.Fatalf("Download error: %v", err)
	}
	data, _ := os.ReadFile(partial)
	if string(data) != "headtail" {
		t.Errorf("resumed content = %q, want headtail", string(data))
	}
}

func TestDownload_FileNotFoundStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/models/test-model/tree/main" {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`[{"path":"missing.safetensors"}]`))
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	d := NewHuggingFaceDownloader("").WithBaseURL(server.URL)
	err := d.Download("test-model", t.TempDir(), nil)
	if err == nil {
		t.Error("expected error for 404 file download")
	}
}

func TestDownload_ListWithToken(t *testing.T) {
	var authHeader string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authHeader = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`[]`))
	}))
	defer server.Close()

	d := NewHuggingFaceDownloader("hf_secret").WithBaseURL(server.URL)
	_ = d.Download("test-model", t.TempDir(), nil)
	if authHeader != "Bearer hf_secret" {
		t.Errorf("Authorization = %q", authHeader)
	}
}

func TestDownload_MkdirAllFailure(t *testing.T) {
	// Force MkdirAll to fail by making outputDir a file.
	tmp := t.TempDir()
	blocker := filepath.Join(tmp, "blocker")
	if err := os.WriteFile(blocker, []byte("x"), 0600); err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/models/test-model/tree/main" {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`[{"path":"subdir/model.safetensors"}]`))
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("content"))
	}))
	defer server.Close()

	d := NewHuggingFaceDownloader("").WithBaseURL(server.URL)
	err := d.Download("test-model", filepath.Join(blocker, "out"), nil)
	if err == nil {
		t.Error("expected error when output dir cannot be created")
	}
}
