package web

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestNewSSEHub(t *testing.T) {
	h := NewSSEHub()
	if h.clients == nil {
		t.Error("clients map is nil")
	}
	if len(h.clients) != 0 {
		t.Errorf("clients len = %d, want 0", len(h.clients))
	}
}

func TestAddClient(t *testing.T) {
	h := NewSSEHub()
	ch := h.AddClient()
	if ch == nil {
		t.Fatal("AddClient returned nil channel")
	}
	if len(h.clients) != 1 {
		t.Errorf("clients len = %d, want 1", len(h.clients))
	}
}

func TestRemoveClient(t *testing.T) {
	h := NewSSEHub()
	ch := h.AddClient()
	h.RemoveClient(ch)

	if len(h.clients) != 0 {
		t.Errorf("clients len after remove = %d, want 0", len(h.clients))
	}

	select {
	case _, ok := <-ch:
		if ok {
			t.Error("channel should be closed after removal")
		}
	default:
		t.Error("channel should be closed")
	}
}

func TestBroadcast(t *testing.T) {
	h := NewSSEHub()
	ch1 := h.AddClient()
	ch2 := h.AddClient()

	h.Broadcast("test-event", "test-data")

	msg1 := <-ch1
	if !strings.Contains(msg1, "event: test-event") {
		t.Errorf("msg1 = %q, want event: test-event", msg1)
	}
	if !strings.Contains(msg1, "data: test-data") {
		t.Errorf("msg1 = %q, want data: test-data", msg1)
	}

	msg2 := <-ch2
	if !strings.Contains(msg2, "event: test-event") {
		t.Errorf("msg2 = %q, want event: test-event", msg2)
	}

	h.RemoveClient(ch1)
	h.RemoveClient(ch2)
}

func TestBroadcast_FullBuffer(t *testing.T) {
	h := NewSSEHub()
	ch := h.AddClient()

	for i := 0; i < 64; i++ {
		h.Broadcast("fill", "data")
	}

	h.Broadcast("overflow", "data")

	for i := 0; i < 64; i++ {
		<-ch
	}

	h.RemoveClient(ch)
}

func TestBroadcast_NoClients(t *testing.T) {
	h := NewSSEHub()
	h.Broadcast("test", "data")
}

func TestServeHTTP_Headers(t *testing.T) {
	h := NewSSEHub()
	ctx, cancel := context.WithCancel(context.Background())
	req := httptest.NewRequest("GET", "/sse", nil).WithContext(ctx)
	w := httptest.NewRecorder()

	done := make(chan bool)
	go func() {
		h.ServeHTTP(w, req)
		done <- true
	}()

	cancel()
	<-done

	hdr := w.Header()
	if hdr.Get("Content-Type") != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", hdr.Get("Content-Type"))
	}
	if hdr.Get("Cache-Control") != "no-cache" {
		t.Errorf("Cache-Control = %q, want no-cache", hdr.Get("Cache-Control"))
	}
	if hdr.Get("Connection") != "keep-alive" {
		t.Errorf("Connection = %q, want keep-alive", hdr.Get("Connection"))
	}
}

func TestServeHTTP_NoFlusher(t *testing.T) {
	h := NewSSEHub()
	req := httptest.NewRequest("GET", "/sse", nil)
	w := &noFlusherWriter{httptest.NewRecorder()}

	h.ServeHTTP(w, req)

	if w.recorder.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", w.recorder.Code)
	}
}

type noFlusherWriter struct {
	recorder *httptest.ResponseRecorder
}

func (w *noFlusherWriter) Header() http.Header {
	return w.recorder.Header()
}

func (w *noFlusherWriter) Write(b []byte) (int, error) {
	return w.recorder.Write(b)
}

func (w *noFlusherWriter) WriteHeader(code int) {
	w.recorder.WriteHeader(code)
}
