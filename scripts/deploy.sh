#!/usr/bin/env bash
# Run on EC2 with IMAGE=<registry>/image-classifier:<commit> bash deploy.sh.
set -eu
: "${IMAGE:?Set IMAGE to the immutable image tag}"
NAME=image-classifier
BACKUP="${NAME}-previous"

if [ -n "${GHCR_TOKEN:-}" ]; then
  (
    DOCKER_CONFIG=$(mktemp -d)
    export DOCKER_CONFIG
    trap 'rm -rf "$DOCKER_CONFIG"' EXIT
    printf '%s' "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USERNAME" --password-stdin
    docker pull "$IMAGE"
  )
else
  docker pull "$IMAGE"
fi
if docker container inspect "$BACKUP" >/dev/null 2>&1; then
  echo "A previous deployment backup exists; inspect it before retrying." >&2
  exit 1
fi

had_previous=false
if docker container inspect "$NAME" >/dev/null 2>&1; then
  docker stop "$NAME"
  docker rename "$NAME" "$BACKUP"
  had_previous=true
fi

rollback() {
  docker logs --tail 50 "$NAME" 2>/dev/null || true
  docker rm -f "$NAME" 2>/dev/null || true
  if [ "$had_previous" = true ]; then
    docker rename "$BACKUP" "$NAME"
    docker start "$NAME"
  fi
}
trap rollback EXIT
docker run -d --name "$NAME" --restart unless-stopped -p 127.0.0.1:8080:8080 "$IMAGE"
for attempt in $(seq 1 30); do
  if [ "$(docker inspect --format '{{.State.Health.Status}}' "$NAME")" = healthy ]; then
    trap - EXIT
    if [ "$had_previous" = true ]; then
      docker rm "$BACKUP"
    fi
    docker ps --filter "name=^/${NAME}$"
    exit 0
  fi
  sleep 5
done
echo "Deployment did not become healthy; restoring the previous container." >&2
exit 1
