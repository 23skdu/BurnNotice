.PHONY: build test lint templ proto clean

build:
	go build -o bin/burnnotice ./cmd/burnnotice
	go build -o bin/dashboard ./cmd/dashboard

test:
	go test ./... -v -race

lint:
	golangci-lint run ./...

templ:
	templ generate

proto:
	protoc --go_out=. --go-grpc_out=. proto/*.proto

clean:
	rm -rf bin/
	rm -f templates/*_templ.go
	go clean -cache
