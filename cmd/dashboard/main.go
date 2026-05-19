package main

import (
	"log"
	"net/http"
	"time"
)

func main() {
	mux := http.NewServeMux()

	// TODO: Wire up handlers from internal/web
	// mux.Handle("GET /{$}", h.Dashboard())
	// mux.Handle("GET /profiler", h.Profiler())
	// mux.Handle("GET /model/{id}", h.ModelProfile())
	// mux.Handle("GET /monitor/{session}", h.Monitor())
	// mux.Handle("GET /sse", h.SSE())

	mux.Handle("GET /static/", http.StripPrefix("/static/", http.FileServer(http.Dir("static"))))

	srv := &http.Server{
		Addr:         ":8080",
		Handler:      mux,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	log.Println("Dashboard listening on :8080")
	log.Fatal(srv.ListenAndServe())
}
