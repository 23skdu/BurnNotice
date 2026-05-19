package web

import (
	"fmt"
	"net/http"
	"sync"

	"burnnotice/internal/metrics"
)

// SSEHub manages real-time event broadcasting to connected clients.
type SSEHub struct {
	mu      sync.RWMutex
	clients map[chan string]bool
}

// NewSSEHub creates a new SSE hub.
func NewSSEHub() *SSEHub {
	return &SSEHub{
		clients: make(map[chan string]bool),
	}
}

// AddClient registers a new client channel.
func (h *SSEHub) AddClient() chan string {
	h.mu.Lock()
	defer h.mu.Unlock()
	ch := make(chan string, 64)
	h.clients[ch] = true
	metrics.SSEConnectionsActive.Set(float64(len(h.clients)))
	return ch
}

// RemoveClient unregisters and closes a client channel.
func (h *SSEHub) RemoveClient(ch chan string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.clients, ch)
	close(ch)
	metrics.SSEConnectionsActive.Set(float64(len(h.clients)))
}

// Broadcast sends an event to all connected clients.
func (h *SSEHub) Broadcast(event, data string) {
	msg := fmt.Sprintf("event: %s\ndata: %s\n\n", event, data)
	h.mu.RLock()
	defer h.mu.RUnlock()
	for ch := range h.clients {
		select {
		case ch <- msg:
			metrics.SSEMessagesSentTotal.Inc()
		default:
			// Client buffer full, skip
		}
	}
}

// ServeHTTP handles SSE connections from browsers.
func (h *SSEHub) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	ch := h.AddClient()
	defer h.RemoveClient(ch)

	for {
		select {
		case <-r.Context().Done():
			return
		case msg := <-ch:
			fmt.Fprint(w, msg)
			flusher.Flush()
		}
	}
}
