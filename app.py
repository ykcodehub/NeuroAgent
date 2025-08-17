import os, io, json, uuid, time, base64, datetime as dt
import streamlit as st
from dotenv import load_dotenv
from pydub import AudioSegment
import pandas as pd

from utils import wx_text_generate, naive_summary, naive_action_items, MiniRAG, safe_json_loads, ibm_stt_transcribe
import prompts as P

load_dotenv()

st.set_page_config(page_title="Team WatsonXoverload - Meeting Agent", layout="wide")
st.title("🤖 Team WatsonXoverload — AI-Powered Meeting Agent (Hinglish + English)")
st.caption("Pre • During • Post | MoM | Action Items | RAG | Extras")

# --- Sidebar: Config ---
with st.sidebar:
    st.header("⚙️ Config")
    WATSONX_API_KEY = st.text_input("WATSONX_API_KEY", os.getenv("WATSONX_API_KEY"), type="password")
    WATSONX_PROJECT_ID = st.text_input("WATSONX_PROJECT_ID", os.getenv("WATSONX_PROJECT_ID"))
    WATSONX_BASE_URL = st.text_input("WATSONX_BASE_URL", os.getenv("WATSONX_BASE_URL", "https://us-south.ml.cloud.ibm.com"))
    WATSONX_MODEL_ID = st.text_input("WATSONX_MODEL_ID", os.getenv("WATSONX_MODEL_ID", "ibm/granite-13b-chat-v2"))

    STT_URL = st.text_input("SPEECH_TO_TEXT_URL", os.getenv("SPEECH_TO_TEXT_URL", ""))
    STT_APIKEY = st.text_input("SPEECH_TO_TEXT_APIKEY", os.getenv("SPEECH_TO_TEXT_APIKEY", ""), type="password")

    SLACK_WEBHOOK_URL = st.text_input("SLACK_WEBHOOK_URL", os.getenv("SLACK_WEBHOOK_URL", ""))

    st.markdown("---")
    st.write("**LLM Mode:**", "watsonx" if WATSONX_API_KEY and WATSONX_PROJECT_ID else "fallback")

# --- RAG documents ---
st.subheader("📚 Reference Docs (for smarter summaries & Q&A)")
rag = MiniRAG()
rag_files = st.file_uploader("Upload reference files (txt/markdown)", type=["txt","md"], accept_multiple_files=True)
if rag_files:
    for f in rag_files:
        text = f.read().decode("utf-8", errors="ignore")
        rag.add(text, {"name": f.name})
    rag.build()
    st.success(f"Loaded {len(rag_files)} docs into mini-RAG.")

# --- Meeting Inputs ---
# app.py (edited section for meeting inputs)

st.subheader("🎙️ Meeting Inputs")
col1, col2 = st.columns(2)

with col1:
    # Option 1: Upload audio file
    audio = st.file_uploader("Upload meeting audio (wav/mp3)", type=["wav", "mp3"])

    # Option 2: Record audio directly
    st.markdown("Or record live audio:")
    recorded_audio = st.audio_input("🎤 Record from microphone")

    lang_model = st.selectbox(
        "STT language model",
        [
            "en-US_Multimedia",   # English (US)
            "hi-IN_Telephony",    # Hindi
            "en-IN_Telephony",    # Indian English
            "es-ES_Multimedia",   # Spanish
            "fr-FR_Multimedia",   # French
            "de-DE_Multimedia",   # German
            "ja-JP_Multimedia",   # Japanese
            "ko-KR_Multimedia",   # Korean
            "ru-RU_Multimedia",   # Russian
            "pt-BR_Multimedia"    # Portuguese (Brazil)
        ]
    )

    if st.button("Transcribe with IBM STT", disabled=not ((audio or recorded_audio) and STT_URL and STT_APIKEY)):
        if audio:
            content = audio.read()
            # Convert mp3 to wav if needed
            if audio.type == "audio/mpeg":
                seg = AudioSegment.from_file(io.BytesIO(content), format="mp3")
                buf = io.BytesIO()
                seg.export(buf, format="wav")
                content = buf.getvalue()
        else:
            # Use recorded audio directly
            content = recorded_audio.getvalue()

        transcript = ibm_stt_transcribe(content, STT_URL, STT_APIKEY, model=lang_model)
        st.session_state["transcript"] = transcript
        st.success("Transcription complete.")



with col2:
    default_sample = "Ravi: Humko project timeline ko 2 hafte extend karna padega..."
    transcript_text = st.text_area("Or paste transcript (Hinglish+English)", value=st.session_state.get("transcript", default_sample), height=220)
    participants = st.text_input("Participants (comma-separated)", "Ravi (PM), Priya (Client), Rahul (Finance), Ankit (Dev)")
    roles = st.text_input("Roles for role-based summaries", "Manager, Developer, Finance, Client")
    hourly_rate = st.number_input("Avg hourly rate per participant (₹)", min_value=0, value=1000, step=100)
    duration_min = st.number_input("Meeting duration (minutes)", min_value=0, value=45, step=5)

# --- Controls ---
st.markdown("---")
colA, colB, colC, colD = st.columns([1,1,1,1])
with colA:
    gen_summary = st.button("🧾 Generate MoM + Action Items")
with colB:
    gen_roles = st.button("👥 Role-based Summaries")
with colC:
    ask_q = st.button("❓ Ask Q&A from Transcript")
with colD:
    analyze_sentiment = st.button("📊 Engagement & Sentiment")

# --- Helpers ---
def call_wx(prompt: str) -> str:
    if WATSONX_API_KEY and WATSONX_PROJECT_ID:
        return wx_text_generate(prompt, WATSONX_MODEL_ID, WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_BASE_URL)
    else:
        return ""

# --- Generate Summaries ---
if gen_summary:
    transcript = transcript_text.strip()
    context_chunks = []
    if rag.docs:
        hits = rag.query(transcript[:500])
        context_chunks = [h["text"][:1000] for h in hits]
    context = "\n\n".join(context_chunks)
    full_input = (P.SYSTEM_SUMMARY + "\n" + P.SUMMARY_PROMPT.format(transcript=transcript + "\n\nContext:\n" + context))
    out = call_wx(full_input)
    if out:
        data = safe_json_loads(out)
    else:
        data = naive_summary(transcript)
    st.session_state["mom"] = data
    st.success("Generated Minutes of Meeting.")

if "mom" in st.session_state:
    st.subheader("🧾 Minutes of Meeting")
    st.json(st.session_state["mom"])
    # Export buttons
    mom_md = ["# Minutes of Meeting",
              f"**Date:** {dt.date.today().isoformat()}",
              f"**Participants:** {participants}",
              "## Key Points"]
    for p in st.session_state["mom"].get("key_points", []):
        mom_md.append(f"- {p}")
    mom_md.append("## Decisions")
    for d in st.session_state["mom"].get("decisions", []):
        mom_md.append(f"- {d}")
    mom_md.append("## Action Items")
    for a in st.session_state["mom"].get("action_items", []):
        if isinstance(a, dict):
            mom_md.append(f"- {a.get('task')} → {a.get('owner','')} (Due: {a.get('due','')})")
        else:
            mom_md.append(f"- {a}")
    mom_markdown = "\n".join(mom_md)
    st.download_button("⬇️ Download MoM.md", mom_markdown, file_name="MoM.md")
    # Action items CSV
    rows = st.session_state["mom"].get("action_items", []) or naive_action_items(transcript_text)
    if rows and isinstance(rows[0], dict):
        df = pd.DataFrame(rows)
        st.download_button("⬇️ Download ActionItems.csv", df.to_csv(index=False), file_name="ActionItems.csv")

# --- Role-based summaries ---
if gen_roles:
    transcript = transcript_text.strip()
    prompt = P.SYSTEM_SUMMARY + "\n" + P.ROLE_BASED_SUMMARY_PROMPT.format(roles=roles, transcript=transcript)
    out = call_wx(prompt)
    if out:
        st.subheader("👥 Role-based Summaries")
        st.write(out)
    else:
        st.info("LLM not configured. Provide WATSONX credentials for role-based summaries.")

# --- Q&A ---
if ask_q:
    user_q = st.text_input("Type your question about the meeting:", key="qa_q", value="What was decided about budget?")
    if st.button("Ask now", key="qa_go"):
        ctx_parts = [transcript_text, json.dumps(st.session_state.get("mom", {}))]
        if rag.docs:
            hits = rag.query(user_q)
            ctx_parts += [h["text"] for h in hits]
        ctx = "\n\n".join(ctx_parts)[:8000]
        prompt = P.SYSTEM_SUMMARY + "\n" + P.QA_PROMPT.format(context=ctx, question=user_q)
        out = call_wx(prompt) or "LLM not configured. Please set WATSONX keys."
        st.write(out)

# --- Sentiment & Engagement ---
if analyze_sentiment:
    transcript = transcript_text.strip()
    out = call_wx(P.SYSTEM_SUMMARY + "\n" + P.SENTIMENT_PROMPT.format(transcript=transcript))
    if out:
        st.subheader("📊 Engagement & Sentiment")
        st.write(out)
    else:
        # naive heuristic
        pos = sum(w in transcript.lower() for w in ["great","good","thanks","done","agree","ok","thik hai"])
        neg = sum(w in transcript.lower() for w in ["delay","issue","problem","nahi","can't","extend"])
        sentiment = "Positive" if pos>neg else ("Negative" if neg>pos else "Neutral")
        st.write(f"Sentiment (heuristic): **{sentiment}**")

# --- Meeting Cost Calculator ---
st.subheader("💸 Meeting Cost Calculator")
num_participants = max(1, len([p for p in participants.split(",") if p.strip()]))
cost = (duration_min/60.0) * num_participants * hourly_rate
st.metric("Estimated Meeting Cost (₹)", f"{cost:,.0f}")

# --- Task push stubs ---
st.subheader("📌 Push Action Items")
if st.button("Send to Slack (webhook)"):
    if SLACK_WEBHOOK_URL and "mom" in st.session_state:
        import requests
        text = "*Action Items*\n" + "\n".join(
            [f"- {a.get('task')} → {a.get('owner','')} (Due: {a.get('due','')})" for a in st.session_state["mom"].get("action_items", []) if isinstance(a, dict)]
        )
        try:
            requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=10)
            st.success("Pushed to Slack.")
        except Exception as e:
            st.error(f"Slack push failed: {e}")
    else:
        st.info("Provide SLACK_WEBHOOK_URL and generate MoM first.")

st.markdown("---")
st.caption("Tip: Configure your IBM Cloud credentials in the sidebar for best results. Demo works with fallbacks if not set.")