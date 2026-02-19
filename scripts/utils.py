import os
import time
import requests

def load_api_keys():
    keys = {}
    path = os.path.join(os.path.dirname(__file__), '..', '.api_keys')
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            k, v = line.split('=', 1)
            keys[k.strip()] = v.strip()
    return keys

def request_with_retry(url, params=None, max_retries=3, timeout=30):
    for i in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            print(f"  Retry {i+1}/{max_retries}: {e}")
            if i < max_retries - 1:
                time.sleep(2 ** i)
    raise Exception(f"Failed after {max_retries} retries: {url}")
