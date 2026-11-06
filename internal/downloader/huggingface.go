package downloader

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync"
)

// HuggingFaceDownloader downloads models from the HuggingFace Hub.
type HuggingFaceDownloader struct {
	token   string
	client  *http.Client
	baseURL string
}

// NewHuggingFaceDownloader creates a downloader with optional API token.
func NewHuggingFaceDownloader(token string) *HuggingFaceDownloader {
	return &HuggingFaceDownloader{
		token:   token,
		client:  &http.Client{},
		baseURL: "https://huggingface.co",
	}
}

// WithBaseURL sets a custom base URL (for testing).
func (d *HuggingFaceDownloader) WithBaseURL(url string) *HuggingFaceDownloader {
	d.baseURL = url
	return d
}

// Download fetches all model files matching the include pattern.
func (d *HuggingFaceDownloader) Download(modelID, outputDir string, includePatterns []string) error {
	files, err := d.listModelFiles(modelID)
	if err != nil {
		return fmt.Errorf("list model files: %w", err)
	}

	var wg sync.WaitGroup
	errCh := make(chan error, len(files))

	for _, file := range files {
		if !shouldDownload(file.Path, includePatterns) {
			continue
		}

		wg.Add(1)
		go func(f string) {
			defer wg.Done()
			if err := d.downloadFile(modelID, f, outputDir); err != nil {
				errCh <- fmt.Errorf("download %s: %w", f, err)
			}
		}(file.Path)
	}

	wg.Wait()
	close(errCh)

	var firstErr error
	for err := range errCh {
		if firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

type hfFile struct {
	Path string `json:"path"`
}

func (d *HuggingFaceDownloader) listModelFiles(modelID string) ([]hfFile, error) {
	url := d.baseURL + "/api/models/" + modelID + "/tree/main"
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	if d.token != "" {
		req.Header.Set("Authorization", "Bearer "+d.token)
	}

	resp, err := d.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API returned %d", resp.StatusCode)
	}

	var files []hfFile
	if err := json.NewDecoder(resp.Body).Decode(&files); err != nil {
		return nil, err
	}
	return files, nil
}

func (d *HuggingFaceDownloader) downloadFile(modelID, filePath, outputDir string) error {
	url := d.baseURL + "/" + modelID + "/resolve/main/" + filePath
	dest := filepath.Join(outputDir, filePath)

	if err := os.MkdirAll(filepath.Dir(dest), 0750); err != nil {
		return err
	}

	var startOffset int64
	if info, err := os.Stat(dest); err == nil {
		startOffset = info.Size()
	}

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	if d.token != "" {
		req.Header.Set("Authorization", "Bearer "+d.token)
	}
	if startOffset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", startOffset))
	}

	resp, err := d.client.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		return fmt.Errorf("download returned %d", resp.StatusCode)
	}

	// #nosec G304 -- output dir is controlled by CLI flag
	out, err := os.OpenFile(dest, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0600)
	if err != nil {
		return err
	}
	defer func() { _ = out.Close() }()

	_, err = io.Copy(out, resp.Body)
	return err
}

func shouldDownload(path string, patterns []string) bool {
	if len(patterns) == 0 {
		patterns = []string{".safetensors", ".bin", ".json", ".model"}
	}
	ext := filepath.Ext(path)
	for _, p := range patterns {
		match := p
		// Support both glob ("*.safetensors") and extension (".safetensors") formats
		if len(match) > 0 && match[0] == '*' {
			match = match[1:]
		}
		if ext == match {
			return true
		}
	}
	return false
}
