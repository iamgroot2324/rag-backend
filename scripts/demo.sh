#!/usr/bin/env bash
# Quick end-to-end demo against a running server. Usage: ./scripts/demo.sh sample.txt
set -euo pipefail
BASE="${BASE_URL:-http://localhost:8000}"
FILE="${1:?usage: demo.sh <file.pdf|file.txt>}"
SID="demo-$(date +%s)"

chat() {
  curl -s -X POST "$BASE/api/v1/chat" -H 'Content-Type: application/json' \
    -d "{\"session_id\":\"$SID\",\"message\":\"$1\"}"; echo
}

echo "== Upload (sentence chunking)"
curl -s -F "file=@$FILE" -F strategy=sentence "$BASE/api/v1/documents"; echo
echo "== Question";  chat "Summarise the document in two sentences."
echo "== Follow-up"; chat "What is the most important point you just mentioned?"
echo "== Booking";   chat "Book an interview for me, my name is Sugam"
chat "sugam@example.com"
chat "2030-01-15 at 14:30"
