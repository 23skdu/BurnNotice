FROM golang:1.26.3-alpine3.21 AS builder

RUN apk add --no-cache git make protoc protobuf-dev

WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w -X main.Version=$(git describe --tags --always 2>/dev/null || echo dev)" \
    -o /bin/burnnotice ./cmd/burnnotice

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w -X main.Version=$(git describe --tags --always 2>/dev/null || echo dev)" \
    -o /bin/burnnotice-dashboard ./cmd/dashboard

FROM alpine:3.21

RUN apk add --no-cache ca-certificates tini

RUN addgroup -g 1000 burnnotice && \
    adduser -u 1000 -G burnnotice -s /bin/sh -D burnnotice

COPY --from=builder /bin/burnnotice /bin/burnnotice-dashboard /bin/

RUN mkdir -p /data/profiles /data/models /static /templates && \
    chown -R burnnotice:burnnotice /data

USER burnnotice

EXPOSE 8080

ENTRYPOINT ["/sbin/tini", "--"]

CMD ["/bin/burnnotice-dashboard"]
