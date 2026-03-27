# Build
FROM golang:1.25-alpine3.23 AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build -o orchestrator .

# Runtime
FROM alpine:3.23

WORKDIR /app

COPY --from=builder /app/orchestrator .

CMD ["./orchestrator"]
