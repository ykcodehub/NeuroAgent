# WatsonX Meeting Agent Demo (Hinglish + English)

A ready-to-run Streamlit demo for **Team WatsonXoverload** — an AI meeting agent across **Pre, During, Post** phases. 
Supports **Hinglish + English**, creates **Minutes of Meeting**, extracts **Action Items**, and includes optional extras.

## Features
- Upload audio (or paste transcript) → Transcription via **IBM Speech-to-Text** (optional).
- **LLM Summaries** via **IBM watsonx.ai** (Granite recommended) with role-based summaries.
- **Action item extraction** (LLM or heuristic fallback).
- **Tiny RAG** over uploaded reference docs for smarter summaries & Q&A.
- **Meeting Cost Calculator**.
- **Engagement & Sentiment** (LLM or heuristic fallback).
- **Task push stubs**: Slack webhook, Trello/Asana placeholders.
- Export **MoM.md** and **ActionItems.csv**.

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Env Vars (recommended via .env)
```
WATSONX_API_KEY=...
WATSONX_PROJECT_ID=...
WATSONX_BASE_URL=https://us-south.ml.cloud.ibm.com
WATSONX_MODEL_ID=ibm/granite-13b-chat-v2
SPEECH_TO_TEXT_URL=https://api.us-south.speech-to-text.watson.cloud.ibm.com
SPEECH_TO_TEXT_APIKEY=...
SLACK_WEBHOOK_URL= # optional
```

> If you don't have keys, the app still runs with **local fallbacks** for summaries/action items.