# Prompts used by the Meeting Agent (Hinglish + English)

SYSTEM_SUMMARY = """You are an expert AI Meeting Assistant for Indian teams.
You understand Hinglish + English and produce concise, structured outputs.
Keep tone professional and clear. Limit fluff.
"""

SUMMARY_PROMPT = """
Summarize the meeting transcript (Hinglish + English allowed).
Return a JSON with keys:
  - agenda
  - key_points (list)
  - decisions (list)
  - risks (list)
  - action_items (list of {task, owner, due, priority})
  - next_steps (list)
Transcript:
{transcript}
"""

ROLE_BASED_SUMMARY_PROMPT = """
Create role-based summaries for roles: {roles}.
For each role, provide:
  - overview (2-3 lines)
  - responsibilities
  - their_action_items (subset relevant to the role)
Keep it crisp, Hinglish allowed.
Transcript:
{transcript}
"""

QA_PROMPT = """
From the transcript + summaries + action items, answer the user's question.
If unsure, say you don't have enough info. Keep answer short.
Context:
{context}
Question: {question}
"""

TRANSLATION_PROMPT = """
Translate the following to {target_lang} while keeping names and product terms intact.
Text: {text}
"""

SENTIMENT_PROMPT = """
Classify overall team sentiment as one of: Positive, Neutral, Negative.
Briefly explain why (1 sentence). Use Hinglish if appropriate.
Transcript:
{transcript}
"""