package downloader

import (
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func FuzzShouldDownload(f *testing.F) {
	f.Add("model.safetensors", []byte(".safetensors"))
	f.Add("README.md", []byte(""))

	f.Fuzz(func(t *testing.T, path string, patterns []byte) {
		if len(path) > 200 {
			return
		}
		var pats []string
		if len(patterns) > 0 {
			pats = []string{string(patterns)}
		}
		result := shouldDownload(path, pats)
		_ = result
	})
}

func FuzzDownloaderDownload(f *testing.F) {
	f.Add("test-model", "model.safetensors")

	f.Fuzz(func(t *testing.T, modelID, filePath string) {
		if len(modelID) > 50 || len(filePath) > 50 || modelID == "" || filePath == "" {
			return
		}
		// Sanitize inputs
		if containsSlash(modelID) || containsSlash(filePath) {
			return
		}
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/models/"+modelID+"/tree/main" {
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`[{"path":"` + filePath + `"}]`))
				return
			}
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("content"))
		}))
		defer server.Close()

		d := NewHuggingFaceDownloader("").WithBaseURL(server.URL)
		outputDir := t.TempDir()
		err := d.Download(modelID, outputDir, nil)
		_ = err
		_ = math.Abs(0)
	})
}

func containsSlash(s string) bool {
	for _, c := range s {
		if c == '/' || c == '\\' {
			return true
		}
	}
	return false
}

func FuzzDownloadFile(f *testing.F) {
	f.Add("model.safetensors", "content")

	f.Fuzz(func(t *testing.T, filePath, content string) {
		if len(filePath) > 50 || len(content) > 1000 || filePath == "" {
			return
		}
		if containsSlash(filePath) {
			return
		}
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(content))
		}))
		defer server.Close()

		d := NewHuggingFaceDownloader("").WithBaseURL(server.URL)
		outputDir := t.TempDir()
		err := d.downloadFile("test-model", filePath, outputDir)
		_ = err
		// Verify file exists if download succeeded
		dest := filepath.Join(outputDir, filePath)
		if err == nil {
			if _, statErr := os.Stat(dest); statErr != nil {
				t.Errorf("file not created: %v", statErr)
			}
		}
	})
}
