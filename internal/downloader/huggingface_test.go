package downloader

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestNewHuggingFaceDownloader(t *testing.T) {
	d := NewHuggingFaceDownloader("test-token")
	if d.token != "test-token" {
		t.Errorf("token = %q, want test-token", d.token)
	}
	if d.client == nil {
		t.Error("client is nil")
	}
	if d.baseURL != "https://huggingface.co" {
		t.Errorf("baseURL = %q, want https://huggingface.co", d.baseURL)
	}

	dNoToken := NewHuggingFaceDownloader("")
	if dNoToken.token != "" {
		t.Errorf("empty token = %q, want empty", dNoToken.token)
	}
}

func TestWithBaseURL(t *testing.T) {
	d := NewHuggingFaceDownloader("").WithBaseURL("http://localhost:8080")
	if d.baseURL != "http://localhost:8080" {
		t.Errorf("baseURL = %q, want http://localhost:8080", d.baseURL)
	}
}

func TestShouldDownload(t *testing.T) {
	tests := []struct {
		path     string
		patterns []string
		want     bool
	}{
		{"model.safetensors", nil, true},
		{"model.bin", nil, true},
		{"config.json", nil, true},
		{"tokenizer.model", nil, true},
		{"README.md", nil, false},
		{"model.safetensors", []string{".safetensors"}, true},
		{"model.bin", []string{".safetensors"}, false},
		{"model.gguf", []string{".gguf"}, true},
		{"nested/dir/model.safetensors", nil, true},
	}

	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			got := shouldDownload(tt.path, tt.patterns)
			if got != tt.want {
				t.Errorf("shouldDownload(%q, %v) = %v, want %v", tt.path, tt.patterns, got, tt.want)
			}
		})
	}
}

func TestDownload_ListAPIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	d := NewHuggingFaceDownloader("").WithBaseURL(server.URL)
	err := d.Download("test-model", t.TempDir(), nil)
	if err == nil {
		t.Error("expected error for API failure")
	}
}

func TestDownload_ListAPIInvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("not json"))
	}))
	defer server.Close()

	d := NewHuggingFaceDownloader("").WithBaseURL(server.URL)
	err := d.Download("test-model", t.TempDir(), nil)
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestDownload_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/models/test-model/tree/main" {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`[{"path":"model.safetensors"},{"path":"config.json"},{"path":"README.md"}]`))
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("file content"))
	}))
	defer server.Close()

	d := NewHuggingFaceDownloader("").WithBaseURL(server.URL)
	outputDir := t.TempDir()
	err := d.Download("test-model", outputDir, nil)
	if err != nil {
		t.Fatalf("Download error: %v", err)
	}

	if _, err := os.Stat(filepath.Join(outputDir, "model.safetensors")); err != nil {
		t.Error("model.safetensors not downloaded")
	}
	if _, err := os.Stat(filepath.Join(outputDir, "config.json")); err != nil {
		t.Error("config.json not downloaded")
	}
	if _, err := os.Stat(filepath.Join(outputDir, "README.md")); err == nil {
		t.Error("README.md should not be downloaded (filtered)")
	}
}

func TestDownload_AuthToken(t *testing.T) {
	var authHeader string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authHeader = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`[]`))
	}))
	defer server.Close()

	d := NewHuggingFaceDownloader("hf_test123").WithBaseURL(server.URL)
	_ = d.Download("test-model", t.TempDir(), nil)

	if authHeader != "Bearer hf_test123" {
		t.Errorf("Authorization header = %q, want Bearer hf_test123", authHeader)
	}
}

func TestDownload_EmptyFileList(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`[]`))
	}))
	defer server.Close()

	d := NewHuggingFaceDownloader("").WithBaseURL(server.URL)
	outputDir := t.TempDir()
	err := d.Download("test-model", outputDir, nil)
	if err != nil {
		t.Fatalf("Download error: %v", err)
	}
}

func TestDownload_FileDownloadError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/models/test-model/tree/main" {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`[{"path":"model.safetensors"}]`))
			return
		}
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	d := NewHuggingFaceDownloader("").WithBaseURL(server.URL)
	err := d.Download("test-model", t.TempDir(), nil)
	if err == nil {
		t.Error("expected error for file download failure")
	}
}

func TestDownload_NestedFilePath(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/models/test-model/tree/main" {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`[{"path":"subdir/model.safetensors"}]`))
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("nested content"))
	}))
	defer server.Close()

	d := NewHuggingFaceDownloader("").WithBaseURL(server.URL)
	outputDir := t.TempDir()
	err := d.Download("test-model", outputDir, nil)
	if err != nil {
		t.Fatalf("Download error: %v", err)
	}

	expected := filepath.Join(outputDir, "subdir", "model.safetensors")
	if _, err := os.Stat(expected); err != nil {
		t.Errorf("nested file not downloaded: %v", err)
	}
}

func TestDownload_CustomPatterns(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/models/test-model/tree/main" {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`[{"path":"model.safetensors"},{"path":"model.gguf"}]`))
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("content"))
	}))
	defer server.Close()

	d := NewHuggingFaceDownloader("").WithBaseURL(server.URL)
	outputDir := t.TempDir()
	err := d.Download("test-model", outputDir, []string{".gguf"})
	if err != nil {
		t.Fatalf("Download error: %v", err)
	}

	if _, err := os.Stat(filepath.Join(outputDir, "model.gguf")); err != nil {
		t.Error("model.gguf not downloaded")
	}
	if _, err := os.Stat(filepath.Join(outputDir, "model.safetensors")); err == nil {
		t.Error("model.safetensors should not be downloaded with .gguf filter")
	}
}
