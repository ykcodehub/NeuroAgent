import os, json, time, math, re
import requests
from typing import Dict, Any, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===== IBM watsonx.ai helpers =====
def _iam_token(api_key: str) -> str:
    """
    Exchange IBM Cloud API key for IAM access token.
    """
    r = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        data={"grant_type":"urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    r.raise_for_status()
    return r.json()["access_token"]

def wx_text_generate(prompt: str, model_id: str, api_key: str, project_id: str, base_url: str) -> str:
    """
    Call watsonx.ai text generation endpoint.
    """
    token = _iam_token(api_key)
    url = f"{base_url}/ml/v1/text/generation?version=2023-05-29"
    payload = {
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 600,
            "temperature": 0.2,
            "repetition_penalty": 1.1,
        },
        "model_id": model_id,
        "project_id": project_id
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type":"application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    try:
        return data["results"][0]["generated_text"]
    except Exception:
        return json.dumps(data, indent=2)

# ===== Simple local extractors (fallback when LLM not available) =====
def naive_action_items(transcript: str) -> List[Dict[str, str]]:
    """
    Extract lines that look like action items using simple heuristics.
    """
    items = []
    for line in transcript.splitlines():
        if any(k in line.lower() for k in ["todo", "action", "krna", "karna", "complete", "finish", "prepare", "banaye", "send", "update"]):
            # Try to detect owner (word before ':') or a name token
            owner = None
            if ":" in line:
                owner = line.split(":")[0].strip()
            m = re.search(r"by ([A-Za-z]+|tomorrow|today|next week|\d{4}-\d{2}-\d{2})", line, flags=re.I)
            due = m.group(0)[3:].strip() if m else ""
            items.append({"task": line.strip(), "owner": owner or "", "due": due, "priority": "Medium"})
    return items

def naive_summary(transcript: str) -> Dict[str, Any]:
    points = [l.strip() for l in transcript.splitlines() if len(l.split())>4][:8]
    return {
        "agenda": "Auto-detected from conversation (may be incomplete).",
        "key_points": points[:5],
        "decisions": points[5:8],
        "risks": [],
        "action_items": naive_action_items(transcript),
        "next_steps": []
    }

# ===== Tiny RAG (local) over uploaded reference files =====
class MiniRAG:
    def __init__(self):
        self.docs = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = None

    def add(self, text: str, meta: Dict[str, Any]):
        self.docs.append({"text": text, "meta": meta})

    def build(self):
        corpus = [d["text"] for d in self.docs] or [""]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def query(self, q: str, k: int=3) -> List[Dict[str, Any]]:
        if not self.docs: return []
        qv = self.vectorizer.transform([q])
        sims = cosine_similarity(qv, self.matrix).flatten()
        idx = sims.argsort()[::-1][:k]
        return [{"text": self.docs[i]["text"], "meta": self.docs[i]["meta"], "score": float(sims[i])} for i in idx]

# ===== IBM Speech to Text (optional) =====
def ibm_stt_transcribe(wav_bytes: bytes, stt_url: str, stt_api_key: str, model: str="en-US_Multimedia"):
    """
    Send WAV/MP3 to IBM STT. Caller must ensure correct content-type.
    """
    auth = ("apikey", stt_api_key)
    headers = {"Content-Type": "audio/wav"}
    r = requests.post(f"{stt_url}/v1/recognize?model={model}", headers=headers, data=wav_bytes, auth=auth, timeout=120)
    r.raise_for_status()
    data = r.json()
    lines = []
    for res in data.get("results", []):
        alt = res.get("alternatives", [{}])[0].get("transcript", "")
        lines.append(alt.strip())
    return "\n".join(lines)

# ===== Utility =====
def safe_json_loads(text: str):
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}