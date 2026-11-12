FROM golang:1.27.4-trixie AS builder

RUN apt-get update && apt-get install -y git make protobuf-compiler libprotobuf-dev gcc g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY go.mod go.sum ./
RUN go mod edit -dropreplace=github.com/23skdu/longbow && go mod download

COPY . .

RUN go mod edit -dropreplace=github.com/23skdu/longbow

# Metal target build (usually built on Darwin directly, but providing cross-compile env or assuming execution on Mac host)
RUN CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 \
    go build -buildvcs=false -ldflags="-s -w -X main.Version=$(git describe --tags --always 2>/dev/null || echo dev)" \
    -o /bin/burnnotice ./cmd/burnnotice

RUN CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 \
    go build -buildvcs=false -ldflags="-s -w -X main.Version=$(git describe --tags --always 2>/dev/null || echo dev)" \
    -o /bin/burnnotice-dashboard ./cmd/dashboard

FROM debian:trixie-slim

RUN apt-get update && apt-get install -y ca-certificates tini && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1000 burnnotice && \
    useradd -u 1000 -g burnnotice -s /bin/sh -m burnnotice

COPY --from=builder /bin/burnnotice /bin/burnnotice-dashboard /bin/

RUN mkdir -p /data/profiles /data/models /static /templates && \
    chown -R burnnotice:burnnotice /data

USER burnnotice
EXPOSE 8080
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/burnnotice-dashboard"]
