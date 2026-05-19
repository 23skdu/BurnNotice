package main

import (
	"log"
	"net/http"
	"os"
	"time"

	"burnnotice/internal/web"
)

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	hub := web.NewSSEHub()
	handlers := web.NewHandlers(hub)

	mux := http.NewServeMux()
	handlers.RegisterRoutes(mux, "static")

	handler := web.Middleware(mux)

	srv := &http.Server{
		Addr:         ":" + port,
		Handler:      handler,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	log.Printf("Dashboard listening on :%s", port) // #nosec G706 -- port from env var, safe for logging
	log.Fatal(srv.ListenAndServe())
}
