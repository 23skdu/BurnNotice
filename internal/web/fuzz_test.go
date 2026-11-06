package web

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func FuzzHandlersAndSSE(f *testing.F) {
	f.Add("llama-3-8b", "sess-100", "alert", `{"score":0.9}`)
	f.Add("qwen-2.5-7b", "sess-200", "metric", `{"val":0.1}`)
	f.Add("", "", "", "")

	f.Fuzz(func(t *testing.T, modelID, sessionID, event, data string) {
		if strings.ContainsAny(modelID, "\r\n") || strings.ContainsAny(sessionID, "\r\n") {
			return
		}
		hub := NewSSEHub()
		ch := hub.AddClient()
		hub.Broadcast(event, data)
		<-ch
		hub.RemoveClient(ch)

		h := NewHandlers(hub)
		req := httptest.NewRequest(http.MethodGet, "/model/test", nil)
		req.SetPathValue("id", modelID)
		req.SetPathValue("session", sessionID)

		w := httptest.NewRecorder()
		h.ModelProfile().ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Fatalf("unexpected status: %d", w.Code)
		}

		w2 := httptest.NewRecorder()
		h.Monitor().ServeHTTP(w2, req)
		if w2.Code != http.StatusOK {
			t.Fatalf("unexpected status: %d", w2.Code)
		}

		svg := sampleLatentSpaceSVG(modelID)
		if svg == "" {
			t.Fatal("sampleLatentSpaceSVG returned empty string")
		}
	})
}
