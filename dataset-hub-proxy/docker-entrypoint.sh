#!/bin/sh
set -eu

if [ -z "${BASIC_AUTH_USERNAME:-}" ] || [ -z "${BASIC_AUTH_PASSWORD:-}" ]; then
    echo "BASIC_AUTH_USERNAME and BASIC_AUTH_PASSWORD must be set." >&2
    exit 1
fi

if [ "$BASIC_AUTH_PASSWORD" = "change-me-before-production" ]; then
    echo "Set FRAME_EXTRACTOR_PASSWORD to a strong password in .env before starting." >&2
    exit 1
fi

htpasswd -Bbc /etc/nginx/.htpasswd "$BASIC_AUTH_USERNAME" "$BASIC_AUTH_PASSWORD"