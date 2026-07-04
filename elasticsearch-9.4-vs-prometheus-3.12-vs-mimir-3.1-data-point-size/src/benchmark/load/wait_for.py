"""Usage: uv run wait-for <url> [retries]

Polls a URL until it responds successfully or the retry budget is exhausted.
Needed for engines whose container image has no shell/wget/curl to run a
Docker Compose `healthcheck:` (e.g. grafana/mimir), so `docker compose up
--wait` can't be used — see deploy/docker/docker-compose.yml's comment on
the mimir service.
"""

import sys
import time
import urllib.request


def main() -> None:
    url = sys.argv[1]
    retries = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    for _ in range(retries):
        try:
            urllib.request.urlopen(url, timeout=2)
            print(f"✓ {url} is ready")
            sys.exit(0)
        except Exception:
            time.sleep(2)

    print(f"❌ {url} did not become ready after {retries * 2}s")
    sys.exit(1)


if __name__ == "__main__":
    main()
