#!/usr/bin/env python3
"""Usage: python3 wait_for.py <url> [retries]"""
import sys, time, urllib.request

url     = sys.argv[1]
retries = int(sys.argv[2]) if len(sys.argv) > 2 else 60

for i in range(retries):
    try:
        urllib.request.urlopen(url, timeout=2)
        print(f"✓ {url} is ready")
        sys.exit(0)
    except Exception:
        time.sleep(2)

print(f"❌ {url} did not become ready after {retries * 2}s")
sys.exit(1)
