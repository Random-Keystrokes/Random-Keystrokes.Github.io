# app.py
import os
import base64
import json
import datetime
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Configure via environment variables
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')  # store this securely (not in code)
GITHUB_OWNER = os.environ.get('GITHUB_OWNER')  # e.g. 'Random-Keystrokes'
GITHUB_REPO = os.environ.get('GITHUB_REPO')    # e.g. 'Random-Keystrokes.Github.io'
BRANCH = os.environ.get('GITHUB_BRANCH', 'main')

if not GITHUB_TOKEN or not GITHUB_OWNER or not GITHUB_REPO:
    raise RuntimeError("Set GITHUB_TOKEN, GITHUB_OWNER, GITHUB_REPO env vars")

GITHUB_API = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents"

def commit_file(path, content_bytes, message):
    """
    Uses GitHub Contents API to create/update a file.
    - path: path in repo (e.g. 'data/session-2025-09-20T12-00-00.json')
    - content_bytes: raw bytes to store (will be base64-encoded)
    """
    url = f"{GITHUB_API}/{path}"
    b64 = base64.b64encode(content_bytes).decode('utf-8')
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    # Check if file exists to supply 'sha' for update
    get_resp = requests.get(url + f"?ref={BRANCH}", headers=headers)
    if get_resp.status_code == 200:
        sha = get_resp.json()['sha']
    else:
        sha = None

    payload = {"message": message, "content": b64, "branch": BRANCH}
    if sha:
        payload["sha"] = sha

    resp = requests.put(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()

@app.route('/collect', methods=['POST'])
def collect():
    try:
        data = request.get_json(force=True)
    except:
        return jsonify({"error":"invalid json"}), 400

    # create a filename based on session & timestamp
    ts = datetime.datetime.utcnow().isoformat(timespec='seconds').replace(':','-')
    session = data.get('meta',{}).get('session','unknown')
    filename = f"data/{session}-{ts}.json"
    content_bytes = json.dumps(data, ensure_ascii=False).encode('utf-8')
    message = f"Add keystroke data {session} @ {ts}"

    try:
        result = commit_file(filename, content_bytes, message)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"ok": True, "commit": result.get('commit', {}).get('sha')}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
