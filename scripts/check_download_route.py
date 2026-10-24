#!/usr/bin/env python3
"""Compare a direct CDN read with the platform route; never print signed URLs."""
import time
import urllib.parse
import urllib.request

from fetch_data import FILES

spec = FILES["vlm-pretrain"]
url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{spec['name']}"
with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=20) as response:
    destination = response.url
print("CDN host:", urllib.parse.urlsplit(destination).hostname, flush=True)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
started = time.monotonic()
try:
    request = urllib.request.Request(destination, headers={"Range": "bytes=0-1048575"})
    with opener.open(request, timeout=15) as response:
        payload = response.read(1048576)
    print("Direct CDN bytes/s:", len(payload) / (time.monotonic() - started), flush=True)
except OSError as error:
    print("Direct CDN unavailable:", type(error).__name__, flush=True)
