import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from google.protobuf.duration_pb2 import Duration
from livekit import api, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    CloseReason,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    llm,
    room_io,
    tokenize,
)
from livekit.plugins import deepgram, google, murf, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

import db
import escalation
import health_tools
import knowledge_base

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# ---------------------------------------------------------------------------
# CALL SUCCESS DEFINITION (Day 8, Health Access track)
#
# A call is SUCCESSFUL if the caller received either:
#   1. Safe guidance — a symptom-triage routing decision (classify_symptom_
#      triage was called and returned a level), or a grounded scheme/
#      eligibility answer (search_knowledge_base found a match), or
#   2. An appropriate escalation — create_escalation succeeded (the caller
#      consented and a request was created or updated), or
#   3. A logged appointment request — any of the three Day-9 appointment
#      specialists' (clinic, radiology, or pathology) book_appointment
#      succeeded (see _AppointmentSpecialistBase.book_appointment, which
#      records this onto the SAME main Assistant instance it was handed off
#      from).
#
# Anything else is a FAILED call — not necessarily a bug, just a call that
# didn't reach the above. See db.FAILURE_CATEGORIES for how failures are
# further broken down, and the close handler in my_agent() below for where
# this decision is actually made, once per call, from what Assistant
# recorded during the conversation.
# ---------------------------------------------------------------------------

db.init_db()
knowledge_base.init_kb()

# Spoken immediately when the call connects, before the caller lookup
# resolves — see the opening-greeting flow in my_agent(). This is the
# complete greeting for a first-time caller; a returning caller additionally
# gets a short personalized welcome-back line once the lookup resolves (see
# the `lookup_result` branch below).
GENERIC_OPENING_LEAD = (
    "Hello! I'm HealthMitra, your AI health assistant. How can I help you today?"
)

# Change this prompt to change what your voice agent does.
# See README.md for example prompts (customer support, language tutor, receptionist).
SYSTEM_PROMPT = """
IDENTITY
You are HealthMitra, an empathetic, AI-powered female health assistant. You work to improve healthcare access for the people of India. You act as a knowledgeable, patient community health companion, supporting callers with reliable public health information while always remembering you are an AI, not a replacement for a human doctor.

OBJECTIVES
A successful call efficiently and safely guides the user to the correct health resource or information. Your primary goals are to:

Assess symptoms to appropriately route the caller (home care, local PHC, or hospital).

Assist ASHA workers by providing quick reference data.

Encourage adherence to prescribed medication routines.

Determine basic eligibility for government health schemes and inform the caller of the required documents.

KNOWLEDGE
You possess knowledge of standard Indian public health guidelines, community health worker protocols, general medication adherence practices, and central/state health scheme criteria (such as PM-JAY).
Your knowledge strictly stops at medical diagnosis and individualized treatment. You do not know what specific illness a caller has, nor what specific medicine they should take to cure it.

LANGUAGE

Default Language (CRITICAL): Start every call in English. On every single turn, respond in whatever language the caller's MOST RECENT message was in — not whatever language the conversation has been in so far. If they speak Hindi, reply in Hindi (Devanagari script, per the Script Requirement below); the moment they speak English again, even after a long stretch of Hindi, reply in English again. Re-check this independently on every turn — do not stay in Hindi out of habit, or because your own last few replies were in Hindi, once the caller has moved back to English. Never comment on a language switch — just pivot naturally, every time, in either direction.

Worked example: Caller says "I have a fever" (English) → you reply in English. Caller then says "मुझे बुखार भी है" (Hindi) → you reply in Hindi. Caller then says "should I go to a hospital?" (English again) → you MUST reply in English again here, even though your last two replies were in Hindi — their latest message decides, every single turn, not the conversation's history.

Script Requirement (CRITICAL): Whenever you are responding in Hindi, you must ALWAYS write it using Devanagari script (देवनागरी लिपि) for Hindi words (e.g., "नमस्ते", "स्वास्थ्य", "मदद"). Never use English/Latin alphabet (Hinglish transliteration) to write Hindi, as the Text-to-Speech engine requires proper Devanagari script for accurate pronunciation. Common English terms (like "AI", "PHC", "Aadhaar", "Doctor") should be written either in Devanagari (एआई, पीएचसी, आधार, डॉक्टर) or clean English words if necessary.

Dynamic Code-Mixing: Fluidly adapt to the caller's exact linguistic style in real-time. If a user mixes English words into Hindi, respond in conversational Hindi using Devanagari script for Hindi words and clear English/Devanagari terms for English words. If they just mix a little Hindi into otherwise-English speech, it's fine to stay in English.

Language Switching: If the caller speaks in a different language entirely mid-call (e.g. Marathi, Bengali), pivot to match that too, without commenting on the switch.

Tone & Register: Speak in simple, everyday language, strictly avoiding complex medical jargon. Maintain a warm, respectful, and reassuring tone. When speaking Hindi, use culturally familiar Indian terms seamlessly (e.g., आंगनवाड़ी, पीएचसी, आधार, रुपये).

Gendered Grammar: Because you have a female persona, use feminine pronouns and conjugations for yourself whenever you're speaking in Hindi (e.g., use "मैं कर सकती हूँ" instead of "मैं कर सकता हूँ").

CALLER PROFILE

You have three tools for the caller's profile: lookup_caller, save_caller_profile, and forget_caller. Use them yourself whenever you need to — nothing about the caller is preloaded for you.

lookup_caller is already called for you once at the very start of every call, in the background while your opening line is being spoken — its result is handed to you as part of your instructions for your very next line, so don't call it again just to open the call. If it found a record, welcome that caller back by name and don't ask for it again. If it found nothing, or later in the call you're unsure whether you already have someone's info, call lookup_caller yourself to check before asking.

Before you ever call save_caller_profile, ask the caller's permission first, in whichever language you're currently speaking with them — e.g. in English: "May I remember this for next time?", or in Hindi: "क्या मैं इसे अगली बार के लिए याद रख सकती हूँ?". Only call save_caller_profile with consent_given=True if they clearly agree. If they say no, or don't clearly agree, do not call save_caller_profile at all — acknowledge that and continue the call normally without saving anything. You only need to ask once per call; if they agree, you can save further details you learn later in the same call without asking again.

When you do have consent, call save_caller_profile with short structured values only — e.g. age_band="31-45", ongoing_conditions="diabetes, hypertension", district="Pune". NEVER write a full medical note, symptom description, or long free text into these fields, only short tags. Once you've decided how to route this call, call save_caller_profile again (still only with consent) with last_triage_outcome summarizing the decision in a few words (e.g. "advised PHC visit", "escalated to 108", "home care advised").

FORGET ME

If the caller asks you to forget them, delete their info, or stop remembering them, confirm once, in the current conversation language — English: "Are you sure? This will permanently delete everything saved about you." / Hindi: "क्या आप निश्चित हैं? इससे आपकी सारी सेव की गई जानकारी हमेशा के लिए मिट जाएगी।" — and only call forget_caller if they confirm. After it runs, tell them plainly that their saved profile has been deleted, in a reassuring tone. This request always takes priority — honor it immediately once confirmed, even mid-conversation.

KNOWLEDGE BASE

For questions about specific government health scheme rules, eligibility, coverage amounts, or required documents (e.g. PM-JAY / Ayushman Bharat), call search_knowledge_base with the caller's question before answering — do not rely on your own memory for these specifics, since they change and you must stay grounded in the actual reference material. Only state facts that came back from the search. If nothing relevant is found, say so plainly in the current conversation language — English: "I don't have exact information on that right now, please check with your nearest PHC." / Hindi: "मुझे इसकी सटीक जानकारी अभी उपलब्ध नहीं है, कृपया नजदीकी पीएचसी में पता करें।" — rather than guessing.

SYMPTOM TRIAGE

As soon as the caller has described their main symptom(s), call classify_symptom_triage with what they told you before you tell them whether to manage at home, go to a PHC, or treat it as an emergency — use the tool's routing, don't decide this from your own judgment alone, so every call is routed consistently. Always pass `language` as whichever language you are currently speaking with the caller (per the Default Language rule), so the card on their screen and the suggested line match the call instead of defaulting to Hindi. Turn its suggested_line into your own natural spoken sentence rather than reading it verbatim, and still follow the Escalation Script below immediately for anything that sounds like a red flag, even before the tool returns.

FACILITY LOOKUP

When the caller asks where to go, wants an address, or you've just told them to visit a PHC/hospital, call find_nearby_health_facility. Reuse a district you already have — from earlier in this call, or from lookup_caller's saved profile — without asking again; only ask the caller for their district/city if you genuinely don't have it from any source yet. Speak the facility names and areas naturally in 1-2 short sentences, never as a read-out list, and always mention in your own words whether this is fresh information or from a saved reference list. If it finds nothing for their area, say so plainly and point them to the 104 health helpline or their local ASHA worker — never invent a facility name or address.

HUMAN ESCALATION

You cannot solve everything yourself. Some situations need a human to look at them — that's what create_escalation is for. It does NOT replace anything else in these instructions; it happens IN ADDITION to your normal spoken response.

When to escalate — only these two situations:
1. Red-flag / diagnosis situations: the caller has a red-flag symptom that triggers the Escalation Script below, OR the caller asks you to diagnose them or name a specific medicine (a Hard Refusal). In both cases, first give the required spoken response (the 108/hospital line, or the "I can't name a medication" deflection) — that already-scripted line is the safety-critical part and always comes first, regardless of whether an escalation ticket ever gets created. Then, once you've said it, offer to also log a request so a human reviews the situation.
2. Genuinely unresolved requests: you tried the relevant tool (search_knowledge_base, find_nearby_health_facility) and it found nothing useful, OR the caller directly asks to speak to a real person / human, and you have no way to satisfy that yourself.

Do NOT escalate for routine questions you already answered, mild/home_care triage, or just because a caller sounds mildly unsatisfied — this is for the two situations above only, not a catch-all.

Ask before sharing (CRITICAL, NON-NEGOTIABLE): Before ever calling create_escalation, tell the caller in plain words, in whichever language you're currently speaking, what you want to send to a human — a short summary of what happened, what you already checked, how urgent it seems, their language, and how they'd like to be followed up with (e.g. a callback on this number, or SMS) — and ask their permission. Only call create_escalation with consent_given=True if they clearly agree, exactly like the Consent Before Saving rule for save_caller_profile. If they say no, do not call it — say that's fine and keep helping however you still can.

What NOT to send: never put a password, OTP, PIN, account number, card number, or any other sensitive identifier into what_happened or already_checked — describe the situation qualitatively (e.g. "caller has chest pain and breathlessness" or "caller asked about PM-JAY coverage, not found in knowledge base") rather than repeating back anything sensitive they said. Do not send the full conversation — only this short summary.

Urgency: pass urgency="emergency" for anything matching the Escalation Script red-flags; "high" for a pregnant/infant case or a diagnosis request that's clearly urgent to the caller; "medium" for an unresolved request with real impact (e.g. a scheme the caller needs soon); "low" for a general "would be good for a human to double check" case.

After it returns: tell the caller their reference id exactly as given (they may need to repeat it to a human later), that the request is now open, and an honest next step — a member of the human support team will review it. Say only that it will be REVIEWED, never that they WILL be contacted, called back, or replied to by a specific time — you don't actually know that a callback will happen, only that a human will look at it. If they ask when, say you don't have an exact timeframe. If the tool tells you a similar request was already open and this one just updated it, say that instead of implying a brand-new one was created.

Updating a request already opened THIS call (CRITICAL): If, after you've already created a request this call, the caller says something that should change its urgency or details (e.g. "please call me immediately", or they describe a worse symptom) — call create_escalation AGAIN with the updated urgency/what_happened. Calling it again with the same reason updates the existing request rather than duplicating it. NEVER tell the caller their request's urgency, priority, or details changed unless you actually called create_escalation again and its result confirms the update (it will say UPDATED and state the new urgency) — do not just say "noted as high priority" or similar from memory. If you're not going to call the tool again, don't claim anything changed; instead say plainly that you don't have a way to guarantee an immediate callback, exactly as usual.

OUTBOUND CALLS (proactive reminders & follow-ups)

Most calls are inbound — the caller reached you. Some calls are ones YOU are placing (see backend/scripts/make_outbound_call.py) for a specific reason. For these, your instructions for the very first thing you say will tell you the call_type and any detail, plus the caller profile lookup already run for you — don't call lookup_caller again for this. Outbound calls always open differently from inbound: the person didn't choose to call, so identify yourself and the reason for calling immediately, in 1-2 sentences, before anything else. Always in English first, per the Default Language rule.

- call_type="medication_reminder": Introduce yourself briefly as HealthMitra, an AI health assistant calling with a reminder, then state the reminder clearly using the given detail — e.g. "Hi, this is HealthMitra — just a reminder to take your evening metformin dose." Then ask if they've taken it, or if they have any questions, and continue naturally from there.

- call_type="vaccination_reminder": Same pattern, but for a vaccination due — state which vaccine/dose using the given detail, then ask if they have any questions or would like help finding a facility (offer find_nearby_health_facility if so).

- call_type="triage_followup": You're calling to check in after a PREVIOUS call where you routed this caller somewhere. Introduce yourself, then reference their last_triage_outcome from the caller profile lookup in your own natural words — never read it verbatim — and ask how it went or how they're feeling now. Example, if last_triage_outcome was "advised PHC visit": "Hi, this is HealthMitra — I'm calling to follow up on our last conversation. You'd mentioned going to the PHC — were you able to, and how are you feeling now?" If there's no last_triage_outcome on file, ask generally how they've been feeling since you last spoke instead of referencing anything specific.

- call_type="escalation_followup": You're calling because a human-escalation request you created on a previous call (see HUMAN ESCALATION above) has now been marked resolved. The given detail describes what the request was about and how it was resolved — introduce yourself, mention in your own words that a human has looked into what they'd raised, and reference the detail naturally, then ask if they have any remaining questions. Example: "Hi, this is HealthMitra — I'm calling to follow up on the request you raised with us. [reference detail naturally] — does that resolve it for you, or is there anything else I can help with?"

Whichever type it is, once you've delivered the reason for calling and gotten a response, the rest of the call proceeds exactly like any other — same tools, same Default Language rule, same guardrails — and call end_call per the Ending the Call rule below once it's naturally finished.

Voicemail (CRITICAL): On an outbound call, what picks up isn't always a person. If what you hear after your opening line sounds like an answering machine or voicemail system — a scripted greeting inviting you to leave a message after a tone, mentioning a mailbox, saying the person is unavailable, or the Hindi equivalent (e.g. "कृपया संदेश छोड़ें", "अभी उपलब्ध नहीं है") — call report_voicemail_detected immediately. Then leave ONE short, appropriate message and hang up: for a medication/vaccination reminder, state the reminder briefly and say you'll try again later; for a triage follow-up, say you were checking in and you'll try again later. Do not try to have a conversation with a voicemail greeting, and do not wait through a long greeting before acting — as soon as you're confident it's not live, act. If you're not genuinely confident it's a machine, treat it as a person instead and continue normally — a real but slow or unusually-worded reply is not voicemail.

GUARDRAILS

Consent Before Saving (CRITICAL, NON-NEGOTIABLE): You must NEVER call save_caller_profile without first asking the caller and having them clearly agree. This applies to every field — name, age band, ongoing conditions, triage outcome, language. No exceptions, no judgment calls, even if it seems obviously helpful to remember. If in doubt, don't save.

Consent Before Escalating (CRITICAL, NON-NEGOTIABLE): You must NEVER call create_escalation without first telling the caller what you want to send and having them clearly agree, per HUMAN ESCALATION above. No exceptions.

Hard Refusals (No Diagnosis & No Drugs): You must NEVER diagnose a condition or name a specific prescription drug. If a user asks what medicine to take, deflect smoothly in the current conversation language — English: "I'm an AI assistant and can't name specific medication. Please take what your doctor prescribed, or visit your nearest PHC." / Hindi: "मैं एक एआई असिस्टेंट हूँ और कोई दवा का नाम नहीं बता सकती। कृपया अपने डॉक्टर द्वारा बताई गई दवा लें या नजदीकी पीएचसी (PHC) जाएँ।" You may only mention basic, standard over-the-counter comforts (like ORS / ओआरएस).

Never-Claims: Never claim to be a doctor, nurse, or a human being. If asked, immediately clarify — in whichever language you're currently speaking — that you are an AI assistant.

Grounded Answers Only: For scheme/eligibility questions covered by search_knowledge_base, never state a specific rule, amount, or document requirement that didn't come back from that search — say you're not sure instead of guessing.

Escalation Script (Red-Flags): If the user mentions any red-flag symptoms (chest pain, severe breathlessness, sudden weakness, heavy bleeding, loss of consciousness) OR if it involves a fever in an infant under 3 months/severe symptoms in a pregnant woman, immediately halt the standard flow and say this in whichever language you're currently speaking:
English: "This sounds like a serious medical situation and I'm not a doctor. Please don't delay — call the 108 ambulance service right now, or go to the nearest hospital."
Hindi: "यह एक गंभीर चिकित्सीय स्थिति लग रही है और मैं डॉक्टर नहीं हूँ। कृपया देर न करें। तुरंत 108 एम्बुलेंस सेवा को कॉल करें या नजदीकी अस्पताल जाएँ।"

STYLE

Opening Greeting (CRITICAL): Your very first line ("Hello! I'm HealthMitra, your AI health assistant. How can I help you today?") is already spoken for you in English the instant the call connects — you don't need to say it again.
- If no prior record is found for this caller, that line is the complete greeting — don't add anything else right now, just wait for them to respond.
- If a record is found with a name, you'll be prompted separately (once the lookup resolves) to add a short, natural welcome-back line by name, in English — e.g. "Oh, welcome back, Ramesh!" If last_triage_outcome is also set, briefly reference it and ask if it helped, in your own natural words — do not read the stored value verbatim and do not repeat "how can I help you today" again, since you already asked that. For example, if last_triage_outcome was "advised PHC visit": "Oh, welcome back, Ramesh! Last time we advised a PHC visit — did that help?"
Stay in English until the caller speaks to you in Hindi — then switch to Hindi for the rest of the call, per the Default Language rule above.

Sentence Length & Pace: Keep responses hyper-concise (1 to 3 short sentences maximum) to ensure smooth performance over telephony channels. Voice callers cannot process long paragraphs.

Turn-Taking: End every turn with a single, clear question or prompt to keep the conversation moving. Never ask multiple questions at once.

Ending the Call (CRITICAL): Nothing else hangs up the call for the caller — you must actively end it yourself, every time, or the call just sits there. As soon as the caller says goodbye, thanks you and indicates they're done, or explicitly asks you to end the call or hang up, say a brief, warm farewell in the current conversation language (English: "Take care, goodbye!" / Hindi: "अपना ख्याल रखें, नमस्ते!") and THEN call end_call in that same turn. Never call end_call before saying goodbye, and never skip calling it once you've said goodbye — a farewell that isn't followed by end_call leaves the caller on a dead line.

Handling Silence & Latency: If the caller is silent, gracefully prompt them once in the current conversation language — English: "Hello, can you hear me?" / Hindi: "नमस्ते, क्या आप मुझे सुन पा रहे हैं?". If there is still no response, say a brief goodbye and call end_call per the Ending the Call rule above rather than waiting indefinitely. Avoid filler words that might disrupt the speech-to-text processing pipeline.

APPOINTMENT SPECIALISTS (Day 9 handoff)

There are three separate specialists, each with exactly one job — booking, checking, rescheduling, or cancelling ONE specific kind of appointment. Route to whichever ONE actually matches what the caller needs:

- transfer_to_clinic_specialist: a doctor/OPD consultation visit at a clinic, PHC, or hospital.
- transfer_to_radiology_specialist: an imaging/scan appointment — X-ray, ultrasound, CT scan, MRI, mammogram, or similar.
- transfer_to_pathology_specialist: a lab/diagnostic test appointment — blood test, urine test, biopsy sample collection, or similar (not imaging).

If the caller isn't sure which kind they need (e.g. "the doctor asked me to get some tests done"), ask ONE brief clarifying question yourself first rather than guessing which specialist to route to. Keep handling plain "where is the nearest facility" questions yourself with find_nearby_health_facility — only hand off once it's genuinely about booking or managing one of these three kinds of appointment. Never hand off for symptoms, triage, scheme/eligibility questions, or anything needing human escalation — those stay with you. Before calling any of these tools, briefly tell the caller in your own words, in whichever language you're currently speaking, that you're connecting them to the relevant specialist.
"""


# ---------------------------------------------------------------------------
# Day 9 — Appointment Specialists
#
# Three separate, narrowly-scoped agents the main Assistant can hand off to
# (see Assistant.transfer_to_clinic_specialist / _to_radiology_specialist /
# _to_pathology_specialist below), one per kind of appointment: a clinic/OPD
# visit, an imaging/radiology scan, or a pathology/lab test. Each does
# exactly one job and nothing else — no symptom advice, no triage, no
# scheme/eligibility answers, no human escalation, no interpreting results.
# Anything outside its own scope goes straight back to the main Assistant
# via transfer_back_to_main_assistant, per the APPOINTMENT SPECIALISTS
# section of SYSTEM_PROMPT above. They share their plumbing (handoff/
# hand-back bookkeeping, facility lookup, appointment CRUD) via
# _AppointmentSpecialistBase below — only IDENTITY/SCOPE/GUARDRAILS differ
# per specialist.
# ---------------------------------------------------------------------------

CLINIC_SPECIALIST_PROMPT = """
IDENTITY

You are the Clinic and Appointment Specialist, a focused assistant that HealthMitra (the main AI health assistant) hands callers off to for exactly one purpose: finding a clinic/hospital and booking, checking, rescheduling, or cancelling a DOCTOR/OPD CONSULTATION appointment there. You are still speaking with the same caller, in the same call — they were just connected to you mid-conversation, so never ask them to re-explain what they already told the main assistant; a short context note about what they need is included in your instructions when you take over.

SCOPE (CRITICAL — stay narrow)

You ONLY handle: finding a nearby clinic/hospital, and booking/checking/rescheduling/cancelling a doctor/OPD consultation appointment. You do NOT give symptom advice, triage guidance, medication guidance, scheme/eligibility answers, or handle human escalation, and you do NOT diagnose anything or name medications — you have no tools for any of that. If the caller actually needs an imaging/scan appointment (X-ray, ultrasound, CT, MRI) or a lab/blood test appointment rather than a doctor visit, call transfer_back_to_main_assistant so the main assistant can route them to the right specialist — do not try to book those yourself. The instant the caller asks about something else outside appointment/clinic logistics (symptoms, an emergency, a government scheme, wanting a human, or anything else), also call transfer_back_to_main_assistant immediately rather than attempting to help yourself, even partially. IMPORTANT: this means a NEW thing the caller says TO YOU after this handoff — the chat history you inherit may include an earlier topic (like symptoms) the caller already discussed with the main assistant BEFORE being connected to you; that was already handled and is NOT by itself a reason to hand back. Also call it once the caller's appointment need is fully handled and they have nothing more for you, or if they want to end the call — the main assistant is the one that actually says goodbye and hangs up.

LANGUAGE

Same rule as the main assistant: respond in whatever language the caller's most recent message was in (English or Hindi), re-checked every turn. When speaking Hindi, always use Devanagari script (देवनागरी लिपि), never Hinglish transliteration. Keep sentences short (1-2 sentences) — this is a phone call.

TOOLS

find_nearby_health_facility: use when the caller needs a clinic/hospital name or address before booking, or asks where to go. Reuse a district you already have (from the handoff context or an earlier lookup) rather than asking again if you already have it.

book_appointment: use once you know which facility, and a preferred date (and time, if given) and a short reason (e.g. "follow-up visit", "new complaint"). This LOGS A REQUEST — it is not a live booking system and does not guarantee the facility can actually see them at that time. Always tell the caller honestly that this is a request that's been noted, give them the reference id, and that the facility/clinic still needs to confirm it — never say the appointment is "confirmed" or "booked" as if it were guaranteed.

list_my_appointments: use if the caller asks about a clinic appointment they already requested, or wants to know what's on file for them.

cancel_appointment: use if the caller wants to cancel a request — confirm which one first if they have more than one on file.

GUARDRAILS

Never invent a facility name, address, or appointment confirmation. If find_nearby_health_facility finds nothing, say so plainly and suggest the caller call the facility directly or ask their local ASHA worker — do not guess. Never claim to be a doctor or a human. If the caller describes a medical emergency or red-flag symptom, do not attempt to advise them yourself — call transfer_back_to_main_assistant immediately so the main assistant's safety-critical escalation script can run; don't delay this handoff by continuing the appointment conversation first.
"""

RADIOLOGY_SPECIALIST_PROMPT = """
IDENTITY

You are the Radiology Appointment Specialist, a focused assistant that HealthMitra (the main AI health assistant) hands callers off to for exactly one purpose: finding a facility and booking, checking, rescheduling, or cancelling an IMAGING/SCAN appointment there — X-ray, ultrasound, CT scan, MRI, mammogram, or similar. You are still speaking with the same caller, in the same call — they were just connected to you mid-conversation, so never ask them to re-explain what they already told the main assistant; a short context note about what they need is included in your instructions when you take over.

SCOPE (CRITICAL — stay narrow)

You ONLY handle: finding a nearby facility with imaging services, and booking/checking/rescheduling/cancelling an imaging/scan appointment. You do NOT give symptom advice, triage guidance, medication guidance, scheme/eligibility answers, or handle human escalation, and you do NOT diagnose anything, interpret or explain what a scan might show, or name medications — you have no tools for any of that and no way to actually see any results. If a caller asks what their scan result means, tell them plainly you can't interpret results and they should discuss it with their doctor, then continue with whatever booking task remains, or hand back if there's nothing else. If the caller actually needs a doctor/OPD consultation or a lab/blood test appointment rather than imaging, call transfer_back_to_main_assistant so the main assistant can route them to the right specialist — do not try to book those yourself. The instant the caller asks about something else outside appointment logistics (symptoms, an emergency, a government scheme, wanting a human, or anything else), also call transfer_back_to_main_assistant immediately rather than attempting to help yourself, even partially. IMPORTANT: this means a NEW thing the caller says TO YOU after this handoff — the chat history you inherit may include an earlier topic (like symptoms) the caller already discussed with the main assistant BEFORE being connected to you; that was already handled and is NOT by itself a reason to hand back. Also call it once the caller's appointment need is fully handled and they have nothing more for you, or if they want to end the call — the main assistant is the one that actually says goodbye and hangs up.

LANGUAGE

Same rule as the main assistant: respond in whatever language the caller's most recent message was in (English or Hindi), re-checked every turn. When speaking Hindi, always use Devanagari script (देवनागरी लिपि), never Hinglish transliteration. Keep sentences short (1-2 sentences) — this is a phone call.

TOOLS

find_nearby_health_facility: use when the caller needs a facility name or address before booking, or asks where to go for imaging. Reuse a district you already have (from the handoff context or an earlier lookup) rather than asking again if you already have it.

book_appointment: use once you know which facility, which scan/imaging test is needed (pass it in `reason`, e.g. "chest X-ray", "abdominal ultrasound"), and a preferred date (and time, if given). This LOGS A REQUEST — it is not a live booking system and does not guarantee the facility can actually do the scan at that time. Always tell the caller honestly that this is a request that's been noted, give them the reference id, and that the facility still needs to confirm it — never say it is "confirmed" or "booked" as if guaranteed. Do not assert prep instructions (e.g. fasting) yourself — tell the caller the facility will confirm any preparation needed when they call to confirm.

list_my_appointments: use if the caller asks about an imaging appointment they already requested, or wants to know what's on file for them.

cancel_appointment: use if the caller wants to cancel a request — confirm which one first if they have more than one on file.

GUARDRAILS

Never invent a facility name, address, or appointment confirmation. If find_nearby_health_facility finds nothing, say so plainly and suggest the caller call the facility directly or ask their local ASHA worker — do not guess. Never claim to be a doctor, radiologist, or a human. If the caller describes a medical emergency or red-flag symptom, do not attempt to advise them yourself — call transfer_back_to_main_assistant immediately so the main assistant's safety-critical escalation script can run; don't delay this handoff by continuing the appointment conversation first.
"""

PATHOLOGY_SPECIALIST_PROMPT = """
IDENTITY

You are the Pathology and Lab Test Specialist, a focused assistant that HealthMitra (the main AI health assistant) hands callers off to for exactly one purpose: finding a facility and booking, checking, rescheduling, or cancelling a LAB/DIAGNOSTIC TEST appointment there — blood test, urine test, biopsy sample collection, or similar (not imaging). You are still speaking with the same caller, in the same call — they were just connected to you mid-conversation, so never ask them to re-explain what they already told the main assistant; a short context note about what they need is included in your instructions when you take over.

SCOPE (CRITICAL — stay narrow)

You ONLY handle: finding a nearby facility/lab, and booking/checking/rescheduling/cancelling a lab/diagnostic test appointment. You do NOT give symptom advice, triage guidance, medication guidance, scheme/eligibility answers, or handle human escalation, and you do NOT diagnose anything, interpret or explain what a test result might mean, or name medications — you have no tools for any of that and no way to actually see any results. If a caller asks what their test result means, tell them plainly you can't interpret results and they should discuss it with their doctor, then continue with whatever booking task remains, or hand back if there's nothing else. If the caller actually needs a doctor/OPD consultation or an imaging/scan appointment rather than a lab test, call transfer_back_to_main_assistant so the main assistant can route them to the right specialist — do not try to book those yourself. The instant the caller asks about something else outside appointment logistics (symptoms, an emergency, a government scheme, wanting a human, or anything else), also call transfer_back_to_main_assistant immediately rather than attempting to help yourself, even partially. IMPORTANT: this means a NEW thing the caller says TO YOU after this handoff — the chat history you inherit may include an earlier topic (like symptoms) the caller already discussed with the main assistant BEFORE being connected to you; that was already handled and is NOT by itself a reason to hand back. Also call it once the caller's appointment need is fully handled and they have nothing more for you, or if they want to end the call — the main assistant is the one that actually says goodbye and hangs up.

LANGUAGE

Same rule as the main assistant: respond in whatever language the caller's most recent message was in (English or Hindi), re-checked every turn. When speaking Hindi, always use Devanagari script (देवनागरी लिपि), never Hinglish transliteration. Keep sentences short (1-2 sentences) — this is a phone call.

TOOLS

find_nearby_health_facility: use when the caller needs a facility/lab name or address before booking, or asks where to go for a test. Reuse a district you already have (from the handoff context or an earlier lookup) rather than asking again if you already have it.

book_appointment: use once you know which facility/lab, which test is needed (pass it in `reason`, e.g. "fasting blood sugar", "complete blood count"), and a preferred date (and time, if given). This LOGS A REQUEST — it is not a live booking system and does not guarantee the lab can actually do the test at that time. Always tell the caller honestly that this is a request that's been noted, give them the reference id, and that the facility still needs to confirm it — never say it is "confirmed" or "booked" as if guaranteed. Do not assert prep instructions (e.g. fasting requirements) yourself as medical fact — tell the caller the lab will confirm any preparation needed when they call to confirm.

list_my_appointments: use if the caller asks about a lab test appointment they already requested, or wants to know what's on file for them.

cancel_appointment: use if the caller wants to cancel a request — confirm which one first if they have more than one on file.

GUARDRAILS

Never invent a facility name, address, or appointment confirmation. If find_nearby_health_facility finds nothing, say so plainly and suggest the caller call the facility directly or ask their local ASHA worker — do not guess. Never claim to be a doctor, lab technician, or a human. If the caller describes a medical emergency or red-flag symptom, do not attempt to advise them yourself — call transfer_back_to_main_assistant immediately so the main assistant's safety-critical escalation script can run; don't delay this handoff by continuing the appointment conversation first.
"""


async def _publish_to_room(room: rtc.Room | None, topic: str, payload: dict) -> None:
    """Best-effort push of a structured tool result to the caller's screen
    (if a UI is attached to this room). Shared by Assistant and all
    Day-9 appointment specialists. Never lets a UI publish failure interrupt
    the voice call.
    """
    if room is None:
        return
    try:
        await room.local_participant.send_text(
            json.dumps(payload, ensure_ascii=False), topic=topic
        )
    except Exception:
        logger.warning("Failed to publish %s data to room", topic, exc_info=True)


class Assistant(Agent):
    def __init__(self, user_id: str | None = None) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)
        # Set once the caller's identity is resolved after the room connects
        # (see resolve_caller_user_id / my_agent below).
        self.user_id = user_id
        # Set right after construction in my_agent() — lets tools push
        # structured data (e.g. facility results) to the caller's screen.
        self.room: rtc.Room | None = None
        # Set right after construction in my_agent() — lets end_call() below
        # actually hang up (delete_room), not just stop replying.
        self.job_ctx: JobContext | None = None
        # Outbound-call bookkeeping (see OUTBOUND CALLS in SYSTEM_PROMPT and
        # the outbound branch of my_agent()). None for inbound calls.
        # outbound_outcome starts as "answered" once a call connects, and
        # only report_voicemail_detected() / the immediate-hangup room
        # handler ever override it to something else — my_agent()'s
        # session.on("close") handler persists whatever it ends up as.
        self.outbound_call: OutboundCall | None = None
        self.outbound_outcome: str | None = None
        # Set by end_call() just before it hangs up, so the SIP
        # participant-disconnected handler in my_agent() can tell "we ended
        # this cleanly" apart from "they hung up on us" — both fire the
        # same disconnect event.
        self.agent_ended_call = False
        # Mirrors my_agent()'s current_tts_locale, updated on every final
        # STT transcript (see _on_user_input_transcribed) — "en" or "hi".
        # end_call() below uses this to pick the language of the fallback
        # farewell it speaks itself when the LLM's turn didn't include one.
        self.current_language = "en"
        # Escalation ids created THIS call, keyed by reason (see
        # create_escalation below). db.find_open_escalation can only dedupe
        # by self.user_id, which is None/empty for a browser caller with no
        # phone number — without this, a second create_escalation call in
        # the same conversation would silently open a duplicate instead of
        # updating the one just created. Reset per call since it's instance
        # state, not persisted.
        self.escalation_ids: dict[str, str] = {}
        # Day-8 call-outcome signals — see CALL SUCCESS DEFINITION below and
        # the outcome-determination logic in my_agent()'s close handler.
        # Populated by classify_symptom_triage / search_knowledge_base /
        # create_escalation as the call actually happens; never written to
        # directly from outside those tools.
        self.triage_results: list[dict] = []
        self.kb_hit = False
        self.escalation_success: dict | None = None
        self.escalation_consent_declined = False
        # Day-9 call-outcome signal — set by ClinicAppointmentSpecialist.
        # book_appointment on THIS SAME instance (it's handed the main agent,
        # never a fresh one — see transfer_to_clinic_specialist) once a
        # request is successfully logged. See CALL SUCCESS DEFINITION above.
        self.appointment_booked: dict | None = None
        # Set True only if a function tool raises rather than returning a
        # normal result — see the function_tools_executed handler in
        # my_agent(), which is the only thing that ever sets this.
        self.tool_exception_occurred = False
        # Day-9 handoff bookkeeping. This SAME Assistant instance is reused
        # across a handoff to ClinicAppointmentSpecialist and back — never
        # reconstructed — so all the state above survives the round trip.
        # transfer_back_to_main_assistant() sets this True right before
        # resuming this agent, so on_enter() below knows to continue the
        # conversation naturally instead of either replaying the opening
        # greeting or staying silent.
        self._returning_from_specialist = False

    async def on_enter(self) -> None:
        # Only fires a reply here on the RETURN leg of a handoff — the very
        # first on_enter (at call start) must stay a no-op, since my_agent()
        # already drives the opening greeting itself (see GENERIC_OPENING_LEAD
        # and the lookup_caller flow below), before the room is even connected.
        if not self._returning_from_specialist:
            return
        self._returning_from_specialist = False
        language_name = "HINDI" if self.current_language == "hi" else "ENGLISH"
        self.session.generate_reply(
            instructions=(
                "The clinic and appointment specialist has just handed the "
                "conversation back to you — either their part is done, or the "
                f"caller asked about something outside appointments. The "
                f"caller is CURRENTLY SPEAKING {language_name} — continue IN "
                f"{language_name}. Do not re-greet them or ask them to repeat "
                "anything already said. If there's nothing more pending, "
                "briefly ask how else you can help."
            )
        )

    @function_tool
    async def lookup_caller(self, context: RunContext) -> str:
        """Check whether this caller has been spoken to before and retrieve
        whatever is on file for them — name, preferred language, and any
        saved facts (age band, ongoing conditions, last triage outcome).
        Call this yourself near the start of the call, before you ask the
        caller their name.
        """
        if not self.user_id:
            return "Caller not yet identified — try again in a moment."

        # Run the (blocking) sqlite read in a worker thread so it never
        # stalls the event loop that's also driving real-time audio.
        caller = await asyncio.to_thread(db.get_caller, self.user_id)
        if not caller:
            return "No prior record — this is a first-time caller."

        return (
            f"name={caller['name']!r}, "
            f"language_preference={caller['language_preference']!r}, "
            f"facts={caller['facts']!r}, "
            f"last_interaction={caller['last_interaction']!r}"
        )

    @function_tool
    async def save_caller_profile(
        self,
        context: RunContext,
        consent_given: bool,
        name: str | None = None,
        age_band: str | None = None,
        ongoing_conditions: str | None = None,
        last_triage_outcome: str | None = None,
        language_preference: str | None = None,
        district: str | None = None,
    ) -> str:
        """Save or update what you know about this caller so they can be
        recognized on a future call. Only pass short, structured values —
        never a written-out medical note or long free text.

        HARD RULE: set consent_given=True only if you have already asked the
        caller, in this call, whether you may remember this — and they
        clearly agreed. If they declined, or you haven't asked, do not call
        this tool at all (or pass consent_given=False). Nothing is saved
        unless consent_given is True.

        Args:
            consent_given: Whether the caller has explicitly agreed, in this
                call, to let you remember this information.
            name: The caller's name, if they've shared it.
            age_band: A short age range, e.g. '18-30', '31-45', '46-60', '60+'.
            ongoing_conditions: Short comma-separated condition tags the
                caller has told you about, e.g. 'diabetes, hypertension'.
                Not a sentence or a note.
            last_triage_outcome: A short label for how this call was routed,
                e.g. 'advised PHC visit', 'escalated - red flag symptoms',
                'home care advised'.
            language_preference: The caller's preferred language, e.g.
                'hindi', 'marathi', 'english'.
            district: The caller's district or city, e.g. 'Pune', 'Patna'.
                Saving this lets you look up nearby facilities for them on a
                future call without asking again.
        """
        if not consent_given:
            logger.info("save_caller_profile skipped — no caller consent")
            return "Not saved — caller consent is required before saving anything."

        if not self.user_id:
            logger.warning(
                "save_caller_profile called before caller identity was resolved"
            )
            return "Could not save — caller not yet identified."

        facts = {
            key: value
            for key, value in {
                "age_band": age_band,
                "ongoing_conditions": ongoing_conditions,
                "last_triage_outcome": last_triage_outcome,
                "district": district,
            }.items()
            if value
        }
        await asyncio.to_thread(
            db.upsert_caller,
            self.user_id,
            name=name,
            language_preference=language_preference,
            facts=facts,
        )
        return "Saved."

    @function_tool
    async def forget_caller(self, context: RunContext) -> str:
        """Permanently delete everything saved about this caller — their
        name, language preference, and all saved facts. Only call this
        after the caller has explicitly asked to be forgotten and confirmed
        they understand it's permanent. This cannot be undone.
        """
        if not self.user_id:
            return "Could not delete — caller not yet identified."

        deleted = await asyncio.to_thread(db.delete_caller, self.user_id)
        if deleted:
            return "Deleted. This caller's saved profile no longer exists."
        return "There was no saved profile for this caller to delete."

    @function_tool
    async def end_call(self, context: RunContext) -> None:
        """Call this to actually hang up and end the call. Call it as soon
        as the caller says goodbye, thanks you and indicates they're done,
        or explicitly asks you to end the call / hang up — do not wait for
        them to disconnect on their own, since nothing else does that for
        them.

        IMPORTANT ORDER: say your brief farewell line FIRST (in whatever
        language you're currently speaking), in the same turn, THEN call
        this tool — never call it before you've said goodbye. If you skip
        the farewell, this tool speaks a generic one on your behalf before
        hanging up, so don't rely on that as a substitute for your own
        warmer, context-appropriate goodbye.
        """
        # Deterministic backstop, not just a prompt instruction: confirmed
        # live (2026-08-25) that the model can call this tool with NO
        # accompanying text at all — no farewell, nothing — hanging up on
        # the caller with total silence. Check what this turn's own
        # assistant message(s) actually said; only trust wait_for_playout()
        # (which lets an existing farewell finish playing) if one of them
        # actually reads as a farewell. Otherwise speak a fallback ourselves
        # first, so a goodbye is said on every single call — not just the
        # ones where the model remembered to include one.
        turn_text = "\n".join(
            item.text_content or ""
            for item in context.speech_handle.chat_items
            if isinstance(item, llm.ChatMessage) and item.role == "assistant"
        )
        if _looks_like_farewell(turn_text):
            # Let the farewell that was just generated finish playing before
            # actually hanging up, so it isn't cut off mid-sentence. Must use
            # RunContext.wait_for_playout() here, not the SpeechHandle's own —
            # this tool call is itself part of that same speech handle, so
            # waiting on the handle directly would be a circular self-wait.
            await context.wait_for_playout()
        else:
            fallback = (
                _FALLBACK_FAREWELL_HI
                if self.current_language == "hi"
                else _FALLBACK_FAREWELL_EN
            )
            logger.warning(
                "end_call invoked with no farewell in this turn — speaking "
                "fallback farewell (%r) before hanging up",
                fallback,
            )
            await context.session.say(fallback)

        self.agent_ended_call = True
        if self.job_ctx is not None:
            await self.job_ctx.delete_room()

    @function_tool
    async def report_voicemail_detected(self, context: RunContext) -> str:
        """Call this the moment you recognize that the line you're
        connected to (on an OUTBOUND call) is an answering machine or
        voicemail system, not a live person — e.g. it plays a scripted
        greeting inviting you to leave a message after a tone, mentions a
        mailbox, says the person is unavailable, or similar, in English or
        Hindi. Be reasonably confident before calling this — a slow or
        unusual-sounding human response is not voicemail; if genuinely
        unsure, treat it as a person and continue the conversation normally
        instead.

        After calling this, per the OUTBOUND CALLS section of your
        instructions: say a short, appropriate message for the call_type
        (state the reminder, or that you tried to follow up), mention
        you'll try again later, THEN call end_call as usual.
        """
        if self.outbound_call is None:
            return "Not applicable — this isn't an outbound call."
        self.outbound_outcome = "voicemail"
        return "Recorded. Say your brief message now, then call end_call."

    @function_tool
    async def search_knowledge_base(self, context: RunContext, query: str) -> str:
        """Search the reference knowledge base (scheme rules, eligibility
        criteria, official guidance documents) for information relevant to
        the caller's question. Call this before answering any question about
        specific scheme names, eligibility, coverage, or required documents
        — do not answer those from memory. Ground your answer only in what
        this returns.

        Args:
            query: The caller's question or the topic to search for, in
                plain words (e.g. "PM-JAY eligibility for farmers").
        """
        results = await asyncio.to_thread(knowledge_base.search, query)
        if not results:
            return "No matching information found in the knowledge base."

        self.kb_hit = True
        return "\n\n".join(f"[{r['source']}] {r['text']}" for r in results)

    async def _publish_to_room(self, topic: str, payload: dict) -> None:
        """Best-effort push of a structured tool result to the caller's
        screen (if a UI is attached to this room). Never lets a UI publish
        failure interrupt the voice call.
        """
        await _publish_to_room(self.room, topic, payload)

    @function_tool
    async def classify_symptom_triage(
        self, context: RunContext, symptoms: str, language: str = "en"
    ) -> str:
        """Classify a caller's described symptoms into a triage level
        (emergency / phc / home_care) using a fixed local rule set modeled
        on standard ASHA/PHC referral guidance (chest pain, breathlessness,
        heavy bleeding, high fever, dehydration, etc.). Call this as soon as
        the caller has described their main symptom(s), before you tell
        them whether to treat it as an emergency, go to a PHC, or manage at
        home — use this tool's routing rather than deciding from your own
        judgment, so every call is routed the same, traceable way. Do not
        call this for general health questions that aren't about the
        caller's own current symptoms — use search_knowledge_base for
        scheme/eligibility questions instead.

        Args:
            symptoms: A short description of what the caller told you they
                are experiencing, in their own words (Hindi or English),
                e.g. "बुखार तीन दिन से है और बहुत कमजोरी लग रही है".
            language: "en" or "hi" — whichever language you are CURRENTLY
                speaking with the caller right now (per the Default
                Language rule), not necessarily the language of `symptoms`
                itself. Determines the language of the text shown on the
                caller's screen and the suggested spoken line, so it always
                matches the call rather than defaulting to Hindi.
        """
        result = await asyncio.to_thread(
            health_tools.classify_triage, symptoms, language
        )
        self.triage_results.append(result)
        await self._publish_to_room("healthmitra-triage", result)

        return (
            f"triage_level={result['level']!r}, matched_on={result['matched_keyword']!r}, "
            f"suggested_line={result['advice']!r}, "
            f"source={result['source']} ({result['ruleset_version']}, a fixed local "
            "ruleset, not a live medical database). Say this in your own natural "
            "words, 1-3 short sentences, in the same language it's already written "
            "in — do not read these fields out loud."
        )

    @function_tool
    async def find_nearby_health_facility(
        self, context: RunContext, district: str | None = None
    ) -> str:
        """Look up nearby government health facilities (hospitals/PHCs) for
        a district. Call this whenever the caller asks where to go, wants
        an address, or right after you've told them to visit a PHC/hospital
        and they might want to know which one.

        If you already know the caller's district — from earlier in this
        call, or from their saved profile via lookup_caller — pass it here
        yourself; do NOT ask the caller for their district again if you
        already have it from either source. Only ask the caller directly if
        you truly don't have it yet.

        If this returns no results, tell the caller plainly and point them
        to the 104 health helpline or their local ASHA worker — never
        invent a facility name or address.

        Args:
            district: The caller's district or city name. If omitted, this
                tool tries the district already saved on the caller's
                profile (from an earlier call) before giving up.
        """
        resolved_district = district
        used_saved_district = False
        if not resolved_district and self.user_id:
            # Chain into the Day-4 caller-profile lookup: reuse a district
            # saved on an earlier call instead of asking again.
            caller = await asyncio.to_thread(db.get_caller, self.user_id)
            if caller and caller["facts"].get("district"):
                resolved_district = caller["facts"]["district"]
                used_saved_district = True

        result = await asyncio.to_thread(
            health_tools.find_facilities, resolved_district or ""
        )
        await self._publish_to_room("healthmitra-facility", result)

        if result["status"] == "no_district":
            return (
                "no_district — you don't have a district for this caller from "
                "any source yet. Ask them which district or city they're in, "
                "then call this tool again with that value."
            )

        if result["status"] == "not_found":
            return (
                f"not_found for district={resolved_district!r} as of "
                f"{result['fetched_at']}. Tell the caller plainly that you "
                "don't have facility information for their area and point "
                "them to the 104 health helpline or their local ASHA worker. "
                "Do not invent a facility name or address."
            )

        facility_lines = "; ".join(
            f"{f['name']} ({f['type']}, {f['area']})" for f in result["facilities"]
        )
        freshness = (
            "fetched live just now from OpenStreetMap"
            if result["data_source"] == "live:openstreetmap-nominatim"
            else "from a saved local reference list, which may not be fully up to date"
        )
        chain_note = (
            " (district reused from this caller's saved profile — do not ask them for it again)"
            if used_saved_district
            else ""
        )
        return (
            f"district={resolved_district!r}{chain_note}, "
            f"facilities=[{facility_lines}], {freshness} "
            f"(as of {result['fetched_at']}). Speak this naturally in 1-2 "
            "sentences — name the facility and area, don't read this out as a list."
        )

    @function_tool
    async def create_escalation(
        self,
        context: RunContext,
        consent_given: bool,
        reason: str,
        urgency: str,
        what_happened: str,
        already_checked: str,
        preferred_contact: str,
    ) -> str:
        """Hand this situation off to a human — see HUMAN ESCALATION in your
        instructions for exactly when to call this and what to say first.
        Only ever call this AFTER telling the caller what you plan to send
        and getting their clear agreement.

        HARD RULE: set consent_given=True only if the caller has just
        clearly agreed, in this call, to you sending this summary to a
        human. If they declined, or you haven't asked, do not call this
        tool at all (or pass consent_given=False) — nothing is created or
        sent unless consent_given is True.

        If an open request already exists for this caller and reason —
        including one you opened earlier in THIS SAME call — this updates
        it (raising its urgency if this report is more urgent, and
        appending your new note) instead of creating a duplicate. The
        result tells you which happened (CREATED vs UPDATED) and the
        request's actual current urgency — only tell the caller their
        request's urgency or details changed if you called this tool again
        and the result says UPDATED. Never claim a change you didn't
        actually make by calling this tool.

        Args:
            consent_given: Whether the caller has explicitly agreed, in this
                call, to you sending this summary to a human.
            reason: Either "red_flag_symptom" (red-flag symptom or a
                request for diagnosis/medication you can't give) or
                "unresolved_request" (your tools couldn't resolve their
                need, or they asked for a human directly).
            urgency: One of "low", "medium", "high", "emergency" — see the
                Urgency guidance in HUMAN ESCALATION.
            what_happened: A short, plain-language summary of the situation
                — e.g. "caller has chest pain and breathlessness for the
                last hour". NEVER include a password, OTP, PIN, account
                number, or other sensitive identifier — describe the
                situation, don't repeat sensitive values back.
            already_checked: A short summary of what you already did or
                told them — e.g. "gave 108/hospital guidance per escalation
                script" or "searched knowledge base, no PM-JAY info found
                for this case". Same rule — no sensitive values.
            preferred_contact: How the caller would like to be followed up
                with, in their own words — e.g. "callback on this number",
                "SMS is fine", "no preference". Do not write out a phone
                number here — you already have their number from the call
                itself if it's a phone call.
        """
        if not consent_given:
            logger.info("create_escalation skipped — no caller consent")
            self.escalation_consent_declined = True
            return "Not created — caller consent is required before sending anything to a human."

        if reason not in escalation.ESCALATION_REASONS:
            return (
                f"Invalid reason {reason!r} — must be one of "
                f"{sorted(escalation.ESCALATION_REASONS)}. Not created."
            )
        if urgency not in escalation.URGENCY_LEVELS:
            return (
                f"Invalid urgency {urgency!r} — must be one of "
                f"{escalation.URGENCY_LEVELS}. Not created."
            )

        who = "unidentified caller"
        language = None
        if self.user_id:
            caller = await asyncio.to_thread(db.get_caller, self.user_id)
            name = caller["name"] if caller else None
            language = caller["language_preference"] if caller else None
            # Only phone-shaped user ids (SIP calls) are meaningful contact
            # info for a human to act on — a browser room identity isn't.
            phone = self.user_id if self.user_id.startswith("+") else None
            if name and phone:
                who = f"{name} ({phone})"
            elif name:
                who = name
            elif phone:
                who = phone

        clean_what_happened = escalation.redact(what_happened)
        clean_already_checked = escalation.redact(already_checked)

        # Prefer the request THIS call already opened for this reason, if
        # any — this is what makes dedup work for a browser caller with no
        # phone-number identity to key db.find_open_escalation off of.
        # Only trust it if it's still open (it could in principle have been
        # resolved mid-call via the dashboard).
        existing = None
        in_call_id = self.escalation_ids.get(reason)
        if in_call_id:
            candidate = await asyncio.to_thread(db.get_escalation, in_call_id)
            if candidate and candidate["status"] != "resolved":
                existing = candidate
        if existing is None:
            existing = await asyncio.to_thread(
                db.find_open_escalation, self.user_id or "", reason
            )
        if existing:
            record = await asyncio.to_thread(
                db.bump_existing_escalation,
                existing["id"],
                urgency=urgency,
                additional_note=clean_what_happened,
            )
            notify_status = await asyncio.to_thread(escalation.notify_webhook, record)
            await asyncio.to_thread(
                db.set_escalation_notify_status, record["id"], notify_status
            )
            self.escalation_ids[reason] = record["id"]
            self.escalation_success = {"reason": reason, "urgency": record["urgency"]}
            return (
                f"UPDATED existing open request, reference_id={record['id']!r}, "
                f"urgency={record['urgency']!r}, status={record['status']!r}, "
                f"notify={notify_status!r}. Tell the caller this reference id and "
                "that it updates a request already in progress for them, per "
                "HUMAN ESCALATION — do not imply a brand-new request was made."
            )

        record = await asyncio.to_thread(
            db.create_escalation,
            user_id=self.user_id,
            reason=reason,
            urgency=urgency,
            who=who,
            what_happened=clean_what_happened,
            already_checked=clean_already_checked,
            language=language,
            preferred_contact=preferred_contact,
            notify_status="pending",
        )
        notify_status = await asyncio.to_thread(escalation.notify_webhook, record)
        await asyncio.to_thread(
            db.set_escalation_notify_status, record["id"], notify_status
        )
        self.escalation_ids[reason] = record["id"]
        self.escalation_success = {"reason": reason, "urgency": urgency}
        return (
            f"CREATED, reference_id={record['id']!r}, urgency={urgency!r}, "
            f"status='open', notify={notify_status!r}. Tell the caller this "
            "reference id clearly and the honest next step per HUMAN "
            "ESCALATION — a human will review it, don't promise a specific "
            "reply time."
        )

    async def _start_appointment_specialist_handoff(
        self,
        specialist_cls: type["_AppointmentSpecialistBase"],
        reason: str,
    ) -> tuple[str, Agent] | str:
        """Shared handoff logic for all three Day-9 appointment specialists
        — construction, the failed-handoff fallback, and the deterministic
        current-language directive (see the on_enter comment on
        _AppointmentSpecialistBase for why that's spelled out explicitly
        rather than left for the model to infer). Not itself a tool — each
        transfer_to_*_specialist method below is the actual tool, with its
        own docstring describing when the LLM should call it.
        """
        try:
            specialist = specialist_cls(
                main_agent=self,
                chat_ctx=self.chat_ctx,
                user_id=self.user_id,
                room=self.room,
                current_language=self.current_language,
                handoff_reason=reason,
            )
        except Exception:
            logger.exception(
                "Failed to start %s handoff", specialist_cls.specialist_label
            )
            failure_language = "HINDI" if self.current_language == "hi" else "ENGLISH"
            return (
                f"TRANSFER FAILED — the {specialist_cls.specialist_label} could "
                f"not be started right now. Apologize briefly IN {failure_language} "
                "(the caller's current language) for the trouble, and keep "
                "helping them yourself as best you can (e.g. "
                "find_nearby_health_facility for a facility address, or "
                "create_escalation if this genuinely needs a human). Do not "
                "mention this as a technical error to the caller — just "
                "smoothly continue helping."
            )
        language_name = "HINDI" if self.current_language == "hi" else "ENGLISH"
        return (
            "TRANSFERRING — if you haven't already said so this turn, tell "
            "the caller now, briefly, that you're connecting them to the "
            f"{specialist_cls.specialist_label}, IN {language_name} — that is "
            "the language the caller is currently speaking with you, per the "
            "conversation so far. Do not use the other language for this "
            "line. The specialist is taking over from here and already "
            f"knows why: {reason!r}.",
            specialist,
        )

    @function_tool
    async def transfer_to_clinic_specialist(
        self, context: RunContext, reason: str
    ) -> tuple[str, Agent] | str:
        """Hand the conversation off to the Clinic and Appointment
        Specialist — a separate agent whose only job is finding a clinic and
        booking, checking, rescheduling, or cancelling a DOCTOR/OPD
        CONSULTATION appointment there.

        Call this when the caller wants to actually book/schedule a doctor
        visit, or check/reschedule/cancel one they already requested. Do NOT
        call this for an imaging/scan appointment (use
        transfer_to_radiology_specialist) or a lab/blood test appointment
        (use transfer_to_pathology_specialist). Do NOT call this just to
        look up a nearby facility's name or address — keep using
        find_nearby_health_facility yourself for that. Do NOT call this for
        symptoms, triage, scheme/eligibility questions, or anything needing
        human escalation — you handle those yourself, per your own
        instructions.

        Before calling this tool, tell the caller in your own words, in
        whichever language you're currently speaking with them right now,
        that you're connecting them to the clinic and appointment
        specialist. Match their most recent message's language exactly —
        do not switch languages for this line.

        Args:
            reason: A short note on what the caller needs (e.g. "wants to
                book a follow-up appointment at Sassoon Hospital next
                Tuesday"), passed to the specialist as context so the caller
                doesn't have to repeat themselves.
        """
        return await self._start_appointment_specialist_handoff(
            ClinicAppointmentSpecialist, reason
        )

    @function_tool
    async def transfer_to_radiology_specialist(
        self, context: RunContext, reason: str
    ) -> tuple[str, Agent] | str:
        """Hand the conversation off to the Radiology Appointment Specialist
        — a separate agent whose only job is finding a facility and booking,
        checking, rescheduling, or cancelling an IMAGING/SCAN appointment
        there — X-ray, ultrasound, CT scan, MRI, mammogram, or similar.

        Call this when the caller wants to actually book/schedule an
        imaging/scan appointment, or check/reschedule/cancel one they
        already requested. Do NOT call this for a doctor/OPD consultation
        (use transfer_to_clinic_specialist) or a lab/blood test appointment
        (use transfer_to_pathology_specialist). Do NOT call this just to
        look up a nearby facility's name or address — keep using
        find_nearby_health_facility yourself for that. Do NOT call this for
        symptoms, triage, scheme/eligibility questions, or anything needing
        human escalation — you handle those yourself, per your own
        instructions.

        Before calling this tool, tell the caller in your own words, in
        whichever language you're currently speaking with them right now,
        that you're connecting them to the radiology appointment specialist.
        Match their most recent message's language exactly — do not switch
        languages for this line.

        Args:
            reason: A short note on what the caller needs (e.g. "needs a
                chest X-ray, doctor's referral in hand"), passed to the
                specialist as context so the caller doesn't have to repeat
                themselves.
        """
        return await self._start_appointment_specialist_handoff(
            RadiologyAppointmentSpecialist, reason
        )

    @function_tool
    async def transfer_to_pathology_specialist(
        self, context: RunContext, reason: str
    ) -> tuple[str, Agent] | str:
        """Hand the conversation off to the Pathology and Lab Test
        Specialist — a separate agent whose only job is finding a facility
        and booking, checking, rescheduling, or cancelling a LAB/DIAGNOSTIC
        TEST appointment there — blood test, urine test, biopsy sample
        collection, or similar (not imaging).

        Call this when the caller wants to actually book/schedule a lab
        test appointment, or check/reschedule/cancel one they already
        requested. Do NOT call this for a doctor/OPD consultation (use
        transfer_to_clinic_specialist) or an imaging/scan appointment (use
        transfer_to_radiology_specialist). Do NOT call this just to look up
        a nearby facility's name or address — keep using
        find_nearby_health_facility yourself for that. Do NOT call this for
        symptoms, triage, scheme/eligibility questions, or anything needing
        human escalation — you handle those yourself, per your own
        instructions.

        Before calling this tool, tell the caller in your own words, in
        whichever language you're currently speaking with them right now,
        that you're connecting them to the pathology and lab test
        specialist. Match their most recent message's language exactly —
        do not switch languages for this line.

        Args:
            reason: A short note on what the caller needs (e.g. "needs a
                fasting blood sugar test"), passed to the specialist as
                context so the caller doesn't have to repeat themselves.
        """
        return await self._start_appointment_specialist_handoff(
            PathologyAppointmentSpecialist, reason
        )

    # To add more tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


class _AppointmentSpecialistBase(Agent):
    """Shared plumbing for the three Day-9 appointment specialists — handoff/
    hand-back bookkeeping, facility lookup, and appointment request CRUD.
    Subclasses (ClinicAppointmentSpecialist, RadiologyAppointmentSpecialist,
    PathologyAppointmentSpecialist below) only supply their own narrow
    IDENTITY/SCOPE/GUARDRAILS instructions plus two class attributes:

    - appointment_type: one of db.APPOINTMENT_TYPES, stored on every
      appointment this specialist books, and used to filter
      list_my_appointments so one specialist doesn't surface another's
      requests.
    - specialist_label: human-readable name used in the main Assistant's
      handoff messages and in this agent's own on_enter introduction.
    """

    appointment_type: str = "clinic"
    specialist_label: str = "Appointment Specialist"

    def __init__(
        self,
        *,
        main_agent: Assistant,
        chat_ctx: llm.ChatContext,
        user_id: str | None,
        room: rtc.Room | None,
        current_language: str,
        handoff_reason: str | None = None,
        instructions: str,
    ) -> None:
        super().__init__(instructions=instructions, chat_ctx=chat_ctx)
        # A reference back to the SAME main-agent instance we were handed off
        # from — never a freshly-constructed Assistant() — so transfer_back_
        # to_main_assistant() resumes it with all its Day-4/7/8 state
        # (user_id, escalation_ids, triage_results, kb_hit, ...) intact
        # instead of losing it to a new object. See Assistant.__init__'s
        # comment on _returning_from_specialist for the other half of this.
        self._main_agent = main_agent
        self.user_id = user_id
        self.room = room
        self.current_language = current_language
        self._handoff_reason = handoff_reason

    async def on_enter(self) -> None:
        context_note = (
            f" The ACTUAL, CURRENT reason you were handed off, per the main "
            f"assistant's handoff note, is: {self._handoff_reason!r} — treat "
            "THIS as the caller's live request right now."
            if self._handoff_reason
            else ""
        )
        # Deliberately no ready-made English/Hindi sentence here — an
        # earlier version embedded literal example text for both languages
        # and the model would sometimes just copy the Hindi one verbatim
        # even in an all-English call (confirmed live, 2026-08-26). Stating
        # the actual current language as a plain fact, instead of offering
        # a template to pick from, is what fixed it.
        #
        # The "IMPORTANT" paragraph below fixes a second, separate bug
        # (confirmed live, 2026-08-26): you inherit the FULL prior chat
        # history, which can include an earlier, already-resolved topic
        # (e.g. the caller got headache advice from the main assistant,
        # THEN asked to book a lab test). Without this instruction, the
        # model would see "headache" anywhere in that history and
        # immediately invoke its own SCOPE guardrail ("hand back if the
        # caller mentions symptoms") against that stale mention — bouncing
        # straight back to the main assistant instead of doing the one job
        # it was actually just handed off for, and then getting handed
        # right back here again, in a loop.
        language_name = "HINDI" if self.current_language == "hi" else "ENGLISH"
        self.session.generate_reply(
            instructions=(
                "You have just taken over this call from the main assistant. "
                f"The caller is CURRENTLY SPEAKING {language_name} — introduce "
                f"yourself briefly IN {language_name} (Devanagari script if "
                f"Hindi), as the {self.specialist_label}, per your own "
                f"IDENTITY instructions.{context_note} Then continue "
                "naturally from there — do NOT ask the caller to repeat "
                "what they already told the main assistant, and do NOT "
                "switch languages for this introduction.\n\nIMPORTANT: the "
                "chat history you can see includes everything from BEFORE "
                "this handoff too, possibly other topics (like symptoms) "
                "the caller already discussed with the main assistant — "
                "those were already handled and are NOT a reason to call "
                "transfer_back_to_main_assistant right now. Only hand back "
                "if the caller says something NEW to you, after this point, "
                "that is actually outside your scope."
            )
        )

    @function_tool
    async def find_nearby_health_facility(
        self, context: RunContext, district: str | None = None
    ) -> str:
        """Look up nearby government health facilities (hospitals/PHCs) for
        a district — use this before booking if you don't yet know which
        facility the caller wants an appointment at, or if they ask where
        to go.

        Reuse a district you already know — from the handoff context, or
        the caller's saved profile — without asking again; only ask the
        caller directly if you truly don't have it from any source yet.

        Args:
            district: The caller's district or city name. If omitted, this
                tool tries the district saved on the caller's profile (from
                an earlier call) before giving up.
        """
        resolved_district = district
        used_saved_district = False
        if not resolved_district and self.user_id:
            caller = await asyncio.to_thread(db.get_caller, self.user_id)
            if caller and caller["facts"].get("district"):
                resolved_district = caller["facts"]["district"]
                used_saved_district = True

        result = await asyncio.to_thread(
            health_tools.find_facilities, resolved_district or ""
        )
        await _publish_to_room(self.room, "healthmitra-facility", result)

        if result["status"] == "no_district":
            return (
                "no_district — you don't have a district for this caller from "
                "any source yet. Ask them which district or city they're in, "
                "then call this tool again with that value."
            )

        if result["status"] == "not_found":
            return (
                f"not_found for district={resolved_district!r} as of "
                f"{result['fetched_at']}. Tell the caller plainly that you "
                "don't have facility information for their area — suggest "
                "they call the facility directly or ask their local ASHA "
                "worker. Do not invent a facility name or address."
            )

        facility_lines = "; ".join(
            f"{f['name']} ({f['type']}, {f['area']})" for f in result["facilities"]
        )
        chain_note = (
            " (district reused from this caller's saved profile — do not ask "
            "them for it again)"
            if used_saved_district
            else ""
        )
        return (
            f"district={resolved_district!r}{chain_note}, "
            f"facilities=[{facility_lines}]. Speak this naturally in 1-2 "
            "sentences, then ask which one they'd like to book an appointment "
            "at (if that's why they're here)."
        )

    @function_tool
    async def book_appointment(
        self,
        context: RunContext,
        facility_name: str,
        preferred_date: str,
        preferred_time: str | None = None,
        reason: str | None = None,
    ) -> str:
        """Log an appointment request for the caller at a specific facility.
        This is NOT a live booking system — it records a request, it does
        not guarantee the facility can see them at that time.

        Args:
            facility_name: The clinic/hospital/lab/imaging center name (e.g.
                from find_nearby_health_facility, or one the caller already
                named).
            preferred_date: The date the caller wants, in their own words or
                a normalized form (e.g. "next Tuesday", "2026-09-02").
            preferred_time: The time the caller wants, if given (e.g.
                "morning", "11am").
            reason: What this appointment is for, e.g. "follow-up visit",
                "chest X-ray", "fasting blood sugar test" — whichever
                applies to what you handle. Never a long medical note.
        """
        district = None
        if self.user_id:
            caller = await asyncio.to_thread(db.get_caller, self.user_id)
            if caller:
                district = caller["facts"].get("district")

        record = await asyncio.to_thread(
            db.create_appointment,
            user_id=self.user_id,
            facility_name=facility_name,
            district=district,
            preferred_date=preferred_date,
            preferred_time=preferred_time,
            reason=reason,
            appointment_type=self.appointment_type,
        )
        # Day-9 call-outcome signal — see CALL SUCCESS DEFINITION near the
        # top of this file. Recorded on the MAIN agent instance (not self),
        # since that's the object my_agent()'s close handler actually reads.
        self._main_agent.appointment_booked = {
            "facility": facility_name,
            "appointment_id": record["id"],
            "type": self.appointment_type,
        }
        return (
            f"REQUESTED, reference_id={record['id']!r}, facility={facility_name!r}, "
            f"date={preferred_date!r}, time={preferred_time!r}. Tell the caller "
            "this reference id clearly, and be honest that this is a REQUEST "
            "that's been noted, not a guaranteed confirmed slot — the facility "
            "still needs to confirm it. Never say it is 'confirmed' or "
            "'booked' as if guaranteed."
        )

    @function_tool
    async def list_my_appointments(self, context: RunContext) -> str:
        """List this caller's active (non-cancelled) appointment requests of
        the kind you handle — use when they ask what they have on file, or
        before cancelling one if they have more than one request.
        """
        if not self.user_id:
            return "Could not look up appointments — caller not yet identified."
        appointments = await asyncio.to_thread(
            db.list_appointments_for_caller, self.user_id, self.appointment_type
        )
        if not appointments:
            return "No active appointment requests on file for this caller."
        lines = "; ".join(
            f"id={a['id']}, facility={a['facility_name']!r}, "
            f"date={a['preferred_date']!r}, time={a['preferred_time']!r}"
            for a in appointments
        )
        return f"Active requests: {lines}. Read these out naturally, not as a raw list."

    @function_tool
    async def cancel_appointment(self, context: RunContext, appointment_id: str) -> str:
        """Cancel one of the caller's appointment requests. If they have more
        than one on file, confirm which one (by facility/date) before calling
        this — use list_my_appointments first if you're not sure.

        Args:
            appointment_id: The reference id of the request to cancel.
        """
        record = await asyncio.to_thread(db.cancel_appointment, appointment_id)
        if record is None:
            return (
                f"NOT_FOUND — no appointment request with id {appointment_id!r}. "
                "Tell the caller you couldn't find that reference id, and "
                "offer to look up their active requests instead."
            )
        return f"CANCELLED, reference_id={record['id']!r}. Confirm this to the caller."

    @function_tool
    async def transfer_back_to_main_assistant(
        self, context: RunContext, reason: str
    ) -> tuple[str, Agent]:
        """Hand the conversation back to HealthMitra, the main assistant.

        Call this as soon as: the caller's appointment need is fully handled
        and they have nothing further for you, OR the caller needs a
        DIFFERENT kind of appointment than the one you handle (the main
        assistant will route them to the right specialist), OR the caller
        asks about anything outside appointment logistics (symptoms, an
        emergency, a scheme question, wanting a human, or anything else),
        OR the caller wants to end the call (the main assistant is the one
        that says goodbye and actually hangs up).

        Args:
            reason: A short note on why you're handing back (e.g. "booked
                appointment, nothing further", "caller actually needs a lab
                test, not imaging", or "caller asked about a scheme, outside
                my scope"), passed to the main assistant.
        """
        await self._main_agent.update_chat_ctx(self.chat_ctx)
        self._main_agent._returning_from_specialist = True
        # Sync the language state back too — while this specialist was
        # active, session.current_agent.current_language updates (see
        # _on_user_input_transcribed in my_agent()) only touched THIS
        # instance, not main_agent, so main_agent's copy would otherwise be
        # stale if the caller switched languages mid-specialist.
        self._main_agent.current_language = self.current_language
        language_name = "HINDI" if self.current_language == "hi" else "ENGLISH"
        return (
            "RETURNING — if you haven't already said so this turn, tell the "
            f"caller now, briefly, IN {language_name} (the caller's current "
            "language), that you're connecting them back to the main "
            f"assistant. Reason for the handback: {reason!r}.",
            self._main_agent,
        )


class ClinicAppointmentSpecialist(_AppointmentSpecialistBase):
    """Day 9 specialist — see the module comment above CLINIC_SPECIALIST_PROMPT
    for what this agent is and isn't responsible for.
    """

    appointment_type = "clinic"
    specialist_label = "Clinic and Appointment Specialist"

    def __init__(self, **kwargs) -> None:
        super().__init__(instructions=CLINIC_SPECIALIST_PROMPT, **kwargs)


class RadiologyAppointmentSpecialist(_AppointmentSpecialistBase):
    """Day 9 specialist — see the module comment above RADIOLOGY_SPECIALIST_PROMPT
    for what this agent is and isn't responsible for.
    """

    appointment_type = "radiology"
    specialist_label = "Radiology Appointment Specialist"

    def __init__(self, **kwargs) -> None:
        super().__init__(instructions=RADIOLOGY_SPECIALIST_PROMPT, **kwargs)


class PathologyAppointmentSpecialist(_AppointmentSpecialistBase):
    """Day 9 specialist — see the module comment above PATHOLOGY_SPECIALIST_PROMPT
    for what this agent is and isn't responsible for.
    """

    appointment_type = "pathology"
    specialist_label = "Pathology and Lab Test Specialist"

    def __init__(self, **kwargs) -> None:
        super().__init__(instructions=PATHOLOGY_SPECIALIST_PROMPT, **kwargs)


def resolve_caller_user_id(participant: rtc.RemoteParticipant) -> str:
    """Stable id used to recognize the same caller on a future call.

    Prefers the SIP phone number when this is a telephony call, since that's
    stable across calls; otherwise falls back to the room participant
    identity assigned by the connecting client.
    """
    return participant.attributes.get("sip.phoneNumber") or participant.identity


# Purposeful outbound call types — see scripts/make_outbound_call.py --type.
# A plain outbound call (phone_number only, no call_type) keeps the original
# generic-greeting behavior; these three open with the actual reason for
# calling instead, per the OUTBOUND CALLS section of SYSTEM_PROMPT.
OUTBOUND_CALL_TYPES = frozenset(
    {
        "medication_reminder",
        "vaccination_reminder",
        "triage_followup",
        "escalation_followup",
    }
)

# ---------------------------------------------------------------------------
# Outbound call outcomes & retry policy.
#
# Inbound calls never need this — the caller either connects or they don't
# call at all. Outbound calls have four failure modes a plain answered/not
# distinction can't tell apart, each with a different reason to (or not to)
# try again:
#
#   no_answer         Rang out, nobody picked up. They just weren't near
#                      the phone — try again reasonably soon.
#   busy              Line engaged right now. Likely free again shortly —
#                      shorter retry delay than no_answer.
#   voicemail         Answered by an answering machine, detected by the
#                      LLM from what it hears after the opening line (see
#                      report_voicemail_detected below — there's no SIP-
#                      level signal for this, Twilio's platform AMD isn't
#                      available over raw SIP trunking). We leave a short
#                      message and don't call back again soon — repeated
#                      same-day attempts just fill up their mailbox.
#   immediate_hangup  They answered and hung up within seconds, before any
#                      real exchange — a rejection, not a missed call.
#                      Retrying soon would be intrusive; wait a full day
#                      and only try once more.
#   failed            Anything else (invalid number, trunk/carrier error,
#                      no trunk configured). Retrying the same number
#                      won't fix a bad number — needs a human to look at
#                      it, not an automatic retry.
#   answered          Connected and nothing special was detected — the
#                      call served its purpose (or ran its natural
#                      course). No retry needed.
#
# Retries aren't executed inline in this job — see
# scripts/retry_outbound_calls.py, run periodically (e.g. via cron), which
# queries db.due_outbound_retries() and re-dispatches them.
#
# The rules themselves — per-outcome delay and attempt cap, plus a master
# on/off switch — live in config/retry_policy.json, not here, so they can
# be tuned (or retries turned off entirely) without a code change or
# restart. Currently OFF (`"enabled": false` in that file) — outbound
# outcomes are still recorded and classified as normal, just nothing gets
# auto-retried while it's off.
RETRY_POLICY_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "retry_policy.json"
)

# A SIP participant that disconnects this soon after answering, without us
# having ended the call ourselves, is almost certainly a rejection rather
# than a call that ran its course.
IMMEDIATE_HANGUP_THRESHOLD_SECONDS = 8.0

# Deterministic backstops for the "Ending the Call" / "Handling Silence"
# rules in SYSTEM_PROMPT: the LLM is instructed to say a farewell and call
# end_call in the SAME turn, but this is a prompt-only guarantee — confirmed
# live (2026-08-25, twice) that it can say the farewell and simply not also
# emit the end_call tool call, leaving the room open indefinitely with
# nobody talking. my_agent()'s idle watchdog force-disconnects in two cases:
#
# 1. farewell_grace_seconds — the agent's own turn looked like the scripted
#    farewell (see _looks_like_farewell), so nothing more is expected from
#    the caller. Short, because we already know the call is over.
# 2. idle_hangup_timeout_seconds — a general backstop for total silence on
#    BOTH sides with no farewell at all (e.g. a stuck tool call, dead air).
#    Long enough not to cut off a slow human response to an actual question.
#
# Both durations live in config/call_ending.json, not here, so they can be
# retuned without a code change or restart — see _load_call_ending_config().
CALL_ENDING_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "call_ending.json"
)
_CALL_ENDING_DEFAULTS = {
    "farewell_grace_seconds": 5.0,
    "idle_hangup_timeout_seconds": 20.0,
}


def _load_call_ending_config() -> dict:
    """Re-read on every call rather than cached at import — a config file
    is only worth having if editing it takes effect without a restart. Any
    problem reading it (missing file, bad JSON) fails closed to the
    defaults above rather than crashing an otherwise-fine call.
    """
    try:
        with CALL_ENDING_CONFIG_PATH.open(encoding="utf-8") as f:
            config = json.load(f)
        return {**_CALL_ENDING_DEFAULTS, **config}
    except (OSError, json.JSONDecodeError) as e:
        logger.error(
            "Failed to load call-ending config from %s: %s — using defaults %s",
            CALL_ENDING_CONFIG_PATH,
            e,
            _CALL_ENDING_DEFAULTS,
        )
        return dict(_CALL_ENDING_DEFAULTS)


# The exact strings SYSTEM_PROMPT's "Ending the Call" rule scripts, verbatim
# reproduced by the model in both live tests above. "ख्याल" (not bare
# "नमस्ते") is used for Hindi specifically because "नमस्ते" alone is also
# used in the *separate* "Handling Silence" check-in line ("नमस्ते, क्या आप
# मुझे सुन पा रहे हैं?" — "hello, can you hear me?"), which must NOT trigger
# the short farewell grace period — that line still expects a reply.
_FAREWELL_MARKERS = ("goodbye", "take care", "ख्याल")


def _looks_like_farewell(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _FAREWELL_MARKERS)


# What Assistant.end_call() speaks itself when the model's turn didn't
# include a farewell — the exact lines SYSTEM_PROMPT's "Ending the Call"
# rule scripts, so a caller can't tell the fallback apart from the real one.
_FALLBACK_FAREWELL_EN = "Take care, goodbye!"
_FALLBACK_FAREWELL_HI = "अपना ख्याल रखें, नमस्ते!"


# Standard SIP final-response codes, mapped to our outcome buckets. Verified
# against two real TwirpErrors from this project's own LiveKit SIP service
# (see backend/README.md "Outbound calling" for the probes used to confirm
# these):
#
#   1. A call the FAR END actively rejects (e.g. carrier/account rejection)
#      raises TwirpError with `sip_status_code` / `sip_status` in
#      `.metadata` — a real SIP status forwarded from the other side.
#   2. A call that just rings out (nobody answers before ringing_timeout)
#      raises TwirpError(code="canceled", message="...sip request timed
#      out...") with NO sip_status_code in `.metadata` at all — LiveKit
#      cancels client-side rather than forwarding a SIP status, since the
#      far end never sent one. Metadata-only classification misses this
#      entirely and silently mislabels every ring-timeout as "failed" — a
#      real bug caught by actually placing a no-answer test call, not by
#      inspection.
#
# Anything matching neither pattern — an unrecognized sip_status_code, or
# an error with no metadata and no "timed out" wording (e.g. no trunk
# configured, malformed request) — falls back to "failed".
_SIP_BUSY_CODES = {"486", "600", "603"}  # Busy Here / Busy Everywhere / Decline
_SIP_NO_ANSWER_CODES = {
    "408",
    "480",
    "487",
}  # Timeout / Temporarily Unavailable / Terminated


def _classify_sip_error(error: api.TwirpError) -> str:
    code = (error.metadata or {}).get("sip_status_code")
    if code in _SIP_BUSY_CODES:
        return "busy"
    if code in _SIP_NO_ANSWER_CODES:
        return "no_answer"
    if error.code == "canceled" or "timed out" in error.message.lower():
        return "no_answer"
    return "failed"


def _load_retry_policy() -> dict:
    """Re-read on every call rather than cached at import — a config file
    is only worth having if editing it takes effect without a restart.
    Any problem reading it (missing file, bad JSON) fails closed to "no
    retries" rather than crashing an otherwise-fine outbound call.
    """
    try:
        with RETRY_POLICY_PATH.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.error(
            "Failed to load retry policy from %s: %s — treating retries as disabled",
            RETRY_POLICY_PATH,
            e,
        )
        return {"enabled": False, "outcomes": {}}


def next_retry_at(outcome: str, attempt_number: int) -> float | None:
    """When the NEXT attempt should happen, or None if retries are
    disabled, this outcome doesn't retry, or `attempt_number` has already
    reached this outcome's cap.
    """
    config = _load_retry_policy()
    if not config.get("enabled", False):
        return None
    policy = config.get("outcomes", {}).get(outcome)
    if policy is None or attempt_number >= policy["max_attempts"]:
        return None
    return time.time() + policy["delay_minutes"] * 60


@dataclass(frozen=True)
class OutboundCall:
    phone_number: str
    call_type: str | None  # one of OUTBOUND_CALL_TYPES, or None for a plain call
    detail: (
        str | None
    )  # e.g. "your evening metformin dose" — not used for triage_followup
    attempt_number: (
        int  # 1 for a fresh call; >1 for a retry (see scripts/retry_outbound_calls.py)
    )


def _extract_outbound_call(job_metadata: str) -> OutboundCall | None:
    """A job is an outbound call if its dispatch metadata carries a
    `phone_number` (see scripts/make_outbound_call.py). Anything else —
    empty, malformed, or without that key — is treated as a normal inbound
    call, never as an error. An unrecognized `call_type` is treated the same
    as no call_type (falls back to the plain generic-greeting flow) rather
    than failing the call outright.
    """
    if not job_metadata:
        return None
    try:
        data = json.loads(job_metadata)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    phone_number = data.get("phone_number")
    if not isinstance(phone_number, str) or not phone_number.strip():
        return None

    call_type = data.get("call_type")
    if call_type not in OUTBOUND_CALL_TYPES:
        call_type = None

    detail = data.get("detail")
    detail = detail if isinstance(detail, str) and detail.strip() else None

    attempt_number = data.get("attempt_number")
    attempt_number = (
        attempt_number if isinstance(attempt_number, int) and attempt_number > 0 else 1
    )

    return OutboundCall(
        phone_number=phone_number,
        call_type=call_type,
        detail=detail,
        attempt_number=attempt_number,
    )


async def _dial_outbound_participant(ctx: JobContext, phone_number: str) -> str:
    """Places the outbound call over the Twilio SIP trunk and blocks until
    the callee answers. Returns the outcome ("answered", "no_answer",
    "busy", or "failed") — never raises — so a failed dial ends the job
    cleanly instead of crashing it or trying to talk to an empty room.
    """
    trunk_id = os.environ.get("LIVEKIT_SIP_OUTBOUND_TRUNK_ID")
    if not trunk_id:
        logger.error(
            "Cannot place outbound call to %s — LIVEKIT_SIP_OUTBOUND_TRUNK_ID is not "
            "set (see scripts/setup_outbound_trunk.py)",
            phone_number,
        )
        return "failed"

    try:
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=trunk_id,
                sip_call_to=phone_number,
                participant_identity=f"phone_{phone_number.lstrip('+')}",
                participant_name="Caller",
                # Block here until the call is actually answered, rather
                # than speaking the opening greeting to a ringing line.
                wait_until_answered=True,
                ringing_timeout=Duration(
                    seconds=int(
                        os.environ.get("OUTBOUND_RINGING_TIMEOUT_SECONDS", "30")
                    )
                ),
                max_call_duration=Duration(
                    seconds=int(
                        os.environ.get("OUTBOUND_MAX_CALL_DURATION_SECONDS", "1800")
                    )
                ),
            )
        )
        return "answered"
    except api.TwirpError as e:
        outcome = _classify_sip_error(e)
        logger.warning(
            "Outbound call to %s did not connect (%s): %s", phone_number, outcome, e
        )
        return outcome


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Persist this call and its transcript to SQLite (see src/db.py)
    db_session_id = db.create_call_session(ctx.room.name)

    # Set up a voice AI pipeline using Murf Falcon, Gemini, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        # "multi" enables Deepgram Nova-3's code-switching mode so the same
        # session picks up English (or other supported languages) mid-call
        # instead of forcing every utterance through the Hindi model — a
        # hard "hi" lock here was why switching to English didn't work.
        stt=deepgram.STT(model="nova-3", language="multi"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
            model="gemini-3.5-flash-lite",
        ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        # Starts in English (default language) — the user_input_transcribed
        # handler below switches this to hi-IN once the caller actually
        # speaks Hindi, and back again if they switch back to English.
        tts=murf.TTS(
            voice="Namrita",
            locale="en-IN",
            style="Conversation",
            model="FALCON",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Day-8 call-outcome bookkeeping — see CALL SUCCESS DEFINITION above and
    # _determine_call_outcome() below, which is where these actually get
    # turned into a success/failure verdict once the call ends.
    channel = "browser"  # overwritten by _link_caller() if this is a SIP call
    user_message_count = 0
    latency_samples: list[float] = []  # TTS time-to-first-byte, per agent turn
    last_activity_at = time.time()  # bumped on every turn — see idle watchdog below
    # Set to the timestamp of the agent's turn the moment it looks like the
    # scripted farewell (see _looks_like_farewell); cleared the moment
    # anything else happens. None means "no farewell pending."
    farewell_spoken_at: float | None = None
    # Day-9 handoff-loop visibility (confirmed live, 2026-08-26: a
    # specialist's on_enter reply misread stale, already-resolved context —
    # e.g. an earlier symptom discussion — as a fresh reason to bounce back
    # to the main assistant, which then routed straight back to the same
    # specialist, repeating indefinitely until the caller gave up and
    # disconnected). The actual fix is prompt-level (see the "IMPORTANT" ...
    # not a reason to hand back" text in _AppointmentSpecialistBase.on_enter
    # and each specialist's SCOPE paragraph) — that's a prompt-only
    # guarantee, so this counter is purely a logging backstop to make a
    # recurrence immediately visible in the logs, not a behavioral fix.
    handoffs_since_last_user_message = 0

    @session.on("conversation_item_added")
    def _on_conversation_item_added(event):
        nonlocal user_message_count, last_activity_at, farewell_spoken_at
        nonlocal handoffs_since_last_user_message
        last_activity_at = time.time()
        if event.item.role == "user":
            user_message_count += 1
            handoffs_since_last_user_message = 0
        farewell_spoken_at = (
            last_activity_at
            if event.item.role == "assistant"
            and _looks_like_farewell(event.item.text_content or "")
            else None
        )
        db.save_message(
            db_session_id,
            event.item.role,
            event.item.text_content,
            event.item.created_at,
        )

    handoff_tool_names = frozenset(
        {
            "transfer_to_clinic_specialist",
            "transfer_to_radiology_specialist",
            "transfer_to_pathology_specialist",
            "transfer_back_to_main_assistant",
        }
    )
    handoff_loop_warning_threshold = 3

    @session.on("function_tools_executed")
    def _on_function_tools_executed(event):
        # A tool RAISING (as opposed to returning a normal — possibly
        # "not found" — string) is a genuine tool_failure, not just a call
        # that didn't reach success. See db.FAILURE_CATEGORIES.
        if any(
            output is not None and output.is_error
            for output in event.function_call_outputs
        ):
            assistant.tool_exception_occurred = True

        nonlocal handoffs_since_last_user_message
        if any(call.name in handoff_tool_names for call in event.function_calls):
            handoffs_since_last_user_message += 1
            if handoffs_since_last_user_message == handoff_loop_warning_threshold:
                logger.warning(
                    "Possible Day-9 agent-handoff loop in room %s — %d "
                    "specialist handoffs with no new caller message in "
                    "between",
                    ctx.room.name,
                    handoffs_since_last_user_message,
                )

    def _determine_call_outcome(close_event) -> tuple[str, str | None, str | None]:
        """The CALL SUCCESS DEFINITION, applied to what actually happened
        during this specific call. Returns (outcome, failure_category,
        track_outcome). Only ever called once, from _on_close below.
        """
        if assistant.triage_results:
            level = assistant.triage_results[-1]["level"]
            return "success", None, f"triage:{level}"
        if assistant.escalation_success:
            reason = assistant.escalation_success["reason"]
            urgency = assistant.escalation_success["urgency"]
            return "success", None, f"escalation:{reason}:{urgency}"
        if assistant.kb_hit:
            return "success", None, "kb_answered"
        if assistant.appointment_booked:
            apt_type = assistant.appointment_booked.get("type", "clinic")
            return "success", None, f"appointment_booked:{apt_type}"

        if assistant.tool_exception_occurred:
            return "failed", "tool_failure", None
        if close_event.error is not None or close_event.reason == CloseReason.ERROR:
            return "failed", "api_error", None
        if assistant.escalation_consent_declined:
            return "failed", "user_declined", None
        if user_message_count == 0:
            return "failed", "no_response", None
        if (
            close_event.reason == CloseReason.PARTICIPANT_DISCONNECTED
            and not assistant.agent_ended_call
        ):
            return "failed", "hangup", None
        return "failed", "incomplete", None

    async def _force_disconnect(reason: str) -> None:
        logger.warning("%s — force-disconnecting room %s", reason, ctx.room.name)
        # So _determine_call_outcome doesn't miscategorize this as the
        # caller hanging up on us.
        assistant.agent_ended_call = True
        try:
            await ctx.delete_room()
        except Exception:
            logger.warning("Idle-watchdog force-disconnect failed", exc_info=True)

    async def _idle_hangup_watchdog() -> None:
        """Force-disconnects a call end_call didn't, regardless of whether
        the tool ever actually got invoked — see the two triggers loaded
        from config/call_ending.json below. Read once per call (not on
        every poll) so a mid-call edit to the file can't move the goalposts
        partway through — the next call picks it up instead. Cancelled the
        instant the session closes normally (see _on_close), so it never
        fires against an already-ended call. Polls every second so the
        farewell grace period is honored fairly precisely.
        """
        config = _load_call_ending_config()
        farewell_grace_seconds = config["farewell_grace_seconds"]
        idle_hangup_timeout_seconds = config["idle_hangup_timeout_seconds"]
        try:
            while True:
                await asyncio.sleep(1)
                now = time.time()
                if (
                    farewell_spoken_at is not None
                    and now - farewell_spoken_at >= farewell_grace_seconds
                ):
                    await _force_disconnect(
                        f"Farewell spoken {now - farewell_spoken_at:.0f}s ago with no end_call"
                    )
                    return
                idle_for = now - last_activity_at
                if idle_for >= idle_hangup_timeout_seconds:
                    await _force_disconnect(f"Room idle for {idle_for:.0f}s")
                    return
        except asyncio.CancelledError:
            pass

    @session.on("close")
    def _on_close(event):
        idle_watchdog_task.cancel()
        db.end_call_session(db_session_id, event.reason.value)

        outcome, failure_category, track_outcome = _determine_call_outcome(event)
        avg_latency_ms = (
            round(sum(latency_samples) / len(latency_samples), 1)
            if latency_samples
            else None
        )
        db.record_call_outcome(
            db_session_id,
            outcome=outcome,
            failure_category=failure_category,
            track_outcome=track_outcome,
            channel=channel,
            language="hi" if current_tts_locale.startswith("hi") else "en",
            avg_response_latency_ms=avg_latency_ms,
        )

        # For an outbound call that actually connected, this is where we
        # find out how it ended — persist the final outcome + retry
        # schedule now that we know it. (Dial-time failures — no_answer /
        # busy / failed before anyone answered — are recorded directly in
        # the outbound branch below instead, since there's no session to
        # close in that case.)
        if (
            assistant.outbound_call is not None
            and assistant.outbound_outcome is not None
        ):
            call = assistant.outbound_call
            outcome = assistant.outbound_outcome
            db.record_outbound_attempt(
                phone_number=call.phone_number,
                call_type=call.call_type,
                detail=call.detail,
                attempt_number=call.attempt_number,
                outcome=outcome,
                next_retry_at=next_retry_at(outcome, call.attempt_number),
            )

    # `metrics_collected` fires from a sync callback, so publishing has to be
    # scheduled as a background task — this set holds a strong reference to
    # each one so it can't be garbage-collected mid-flight.
    background_tasks: set[asyncio.Task] = set()

    def _fire_and_forget(coro) -> None:
        task = asyncio.create_task(coro)
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    @session.on("metrics_collected")
    def _on_metrics_collected(event):
        # Surface pipeline latency to the caller's screen (see
        # frontend/components/app/metrics-panel.tsx). EOU transcription
        # delay is the real-world "how long until it heard me" STT latency
        # for a streaming STT — STTMetrics.duration is always 0 for those.
        # TTS ttfb is time-to-first-audio-byte, i.e. "how long until it
        # started speaking."
        metrics = event.metrics
        if metrics.type == "eou_metrics":
            kind, latency_ms = "stt", metrics.transcription_delay * 1000
        elif metrics.type == "tts_metrics":
            kind, latency_ms = "tts", metrics.ttfb * 1000
            # "How long until it started speaking" — the Day-8 dashboard's
            # latency figure. Sampled here rather than STT delay since this
            # is what a caller actually experiences as response time.
            latency_samples.append(latency_ms)
        else:
            return
        payload = json.dumps(
            {
                "kind": kind,
                "latency_ms": round(latency_ms, 1),
                "updated_at": time.time(),
            }
        )
        _fire_and_forget(
            ctx.room.local_participant.send_text(payload, topic="healthmitra-metrics")
        )

    # Tracks which locale the TTS voice is currently speaking in. The
    # session starts in English (see AgentSession(tts=...) above) and only
    # switches to Hindi once the caller actually speaks Hindi — using
    # Deepgram's own per-utterance language detection (STT runs in "multi"
    # code-switching mode) rather than guessing from the LLM's output text.
    current_tts_locale = "en-IN"

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(event):
        nonlocal current_tts_locale
        if not event.is_final or not event.language:
            return
        target_locale = (
            "hi-IN" if str(event.language).lower().startswith("hi") else "en-IN"
        )
        # Mirrored onto whichever agent is CURRENTLY ACTIVE (main assistant
        # or, after a Day-9 handoff, the clinic specialist — both carry a
        # current_language attribute) regardless of whether the TTS locale
        # actually switches below — end_call()'s fallback-farewell language,
        # and the specialist's on_enter introduction, should track the
        # caller's most recent utterance exactly, not the (deliberately
        # stickier) TTS-switch logic below, and not go stale after a handoff.
        session.current_agent.current_language = (
            "hi" if target_locale == "hi-IN" else "en"
        )
        if target_locale != current_tts_locale and session.tts is not None:
            current_tts_locale = target_locale
            # update_options() is a Murf TTS extension, not part of the
            # base plugin interface — swap providers, swap this call too.
            session.tts.update_options(locale=target_locale)

    # Start the session, which initializes the voice pipeline and warms up the models
    assistant = Assistant()
    assistant.room = ctx.room
    assistant.job_ctx = ctx
    await session.start(
        agent=assistant,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    def _link_caller(participant: rtc.RemoteParticipant) -> None:
        # Identity plumbing only — no caller data is read or written here.
        # The agent looks up and saves caller data itself via the
        # lookup_caller / save_caller_profile tools above.
        db.set_participant(db_session_id, participant.identity)
        assistant.user_id = resolve_caller_user_id(participant)
        nonlocal channel
        channel = (
            "sip"
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
            else "browser"
        )

    @ctx.room.on("participant_connected")
    def _on_participant_connected(participant: rtc.RemoteParticipant):
        _link_caller(participant)

    # Join the room and connect to the user
    await ctx.connect()

    # Outbound calls (see scripts/make_outbound_call.py) carry the target
    # phone number — and optionally a purposeful reason to be calling, e.g.
    # a medication/vaccination reminder or a triage follow-up — in the
    # job's dispatch metadata, instead of already having a caller in the
    # room. Dial out and wait for them to answer before falling into a
    # greeting flow.
    outbound_call = _extract_outbound_call(ctx.job.metadata)
    if outbound_call is not None:
        dial_outcome = await _dial_outbound_participant(ctx, outbound_call.phone_number)
        if dial_outcome != "answered":
            # Never answered — no_answer / busy / failed. No session to
            # close, no conversation happened, so record the outcome
            # directly here rather than via _on_close.
            db.record_outbound_attempt(
                phone_number=outbound_call.phone_number,
                call_type=outbound_call.call_type,
                detail=outbound_call.detail,
                attempt_number=outbound_call.attempt_number,
                outcome=dial_outcome,
                next_retry_at=next_retry_at(dial_outcome, outbound_call.attempt_number),
            )
            db.end_call_session(db_session_id, f"outbound_{dial_outcome}")
            # Never connected, so there was never a chance to reach the CALL
            # SUCCESS DEFINITION — record it as a failed call rather than
            # leaving outcome NULL (which would silently exclude it from the
            # Day-8 dashboard's totals).
            db.record_call_outcome(
                db_session_id,
                outcome="failed",
                failure_category="tool_failure"
                if dial_outcome == "failed"
                else "no_response",
                track_outcome=f"outbound_dial:{dial_outcome}",
                channel="sip",
                language=None,
                avg_response_latency_ms=None,
            )
            return

        # Answered — default outcome unless the disconnect handler below or
        # report_voicemail_detected() overrides it before the call ends.
        assistant.outbound_call = outbound_call
        assistant.outbound_outcome = "answered"
        answered_at = time.time()

        def _on_sip_participant_disconnected(
            participant: rtc.RemoteParticipant,
        ) -> None:
            if participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
                return
            # If we hung up ourselves (end_call already ran) or the outcome
            # was already classified some other way (e.g. voicemail), this
            # disconnect is expected — leave the outcome as-is.
            if assistant.agent_ended_call or assistant.outbound_outcome != "answered":
                return
            if time.time() - answered_at < IMMEDIATE_HANGUP_THRESHOLD_SECONDS:
                assistant.outbound_outcome = "immediate_hangup"

        ctx.room.on("participant_disconnected", _on_sip_participant_disconnected)

    # Only start the idle watchdog once the call is actually live — for an
    # outbound call that's right here, after the callee has ANSWERED (the
    # dial_outcome != "answered" branch above already returned), never
    # during the ringing_timeout=30s wait, which would otherwise force-
    # disconnect a call that's still ringing. For inbound, outbound_call is
    # None and this point is reached immediately after connect. Reset the
    # clock right before arming it so nothing counts against the dial delay.
    last_activity_at = time.time()
    idle_watchdog_task = asyncio.create_task(_idle_hangup_watchdog())

    if outbound_call is not None and outbound_call.call_type is not None:
        # Purposeful outbound call: skip the generic inbound-style greeting
        # entirely and open with the actual reason for calling instead (see
        # the OUTBOUND CALLS section of SYSTEM_PROMPT). The callee just
        # answered their phone, so there's no "dead air while ringing" to
        # avoid here the way there is for the inbound generic lead below —
        # awaiting the lookup directly (instead of backgrounding it) is fine.
        for participant in ctx.room.remote_participants.values():
            _link_caller(participant)
            break
        lookup_result = await assistant.lookup_caller(None)
        await session.generate_reply(
            instructions=(
                f"This is an outbound call YOU are placing — call_type="
                f"{outbound_call.call_type!r}, detail={outbound_call.detail!r}. "
                f"Caller profile lookup: {lookup_result}. "
                "Open the call per the OUTBOUND CALLS section of your "
                "instructions for this call_type. Do not call lookup_caller "
                "again for this."
            ),
            allow_interruptions=False,
        )
        return

    # Kick off the caller lookup as soon as we know who's calling, but don't
    # wait on it yet — it runs in the background, concurrently with the
    # opening line below, instead of delaying the start of the call. Covers
    # both inbound calls and a plain outbound call (phone_number only, no
    # call_type) — from here on those two are identical.
    lookup_task: asyncio.Task[str] | None = None
    for participant in ctx.room.remote_participants.values():
        _link_caller(participant)
        lookup_task = asyncio.create_task(assistant.lookup_caller(None))
        break

    # Speak immediately — this doesn't depend on the lookup, so there's no
    # dead air while lookup_task resolves in the background during playback.
    # allow_interruptions=False so an eager caller talking over the greeting
    # can't cut the opening script short before it's said anything useful.
    await session.say(GENERIC_OPENING_LEAD, allow_interruptions=False)

    if lookup_task is not None:
        lookup_result = await lookup_task  # already resolved in practice by now
        # The opening line above is already the complete greeting for a
        # first-time caller — only speak a follow-up when there's an actual
        # name to welcome back, to avoid a redundant "how can I help you
        # today" repeated right after itself.
        if not lookup_result.startswith("No prior record"):
            await session.generate_reply(
                instructions=(
                    "You just spoke your opening line. While you were speaking, "
                    "you already looked up this caller's profile in the "
                    f"background — here's what you found: {lookup_result}. "
                    "Add a short, natural welcome-back line per the Opening "
                    "Greeting rules in your instructions. Do not call "
                    "lookup_caller again for this."
                ),
                allow_interruptions=False,
            )


if __name__ == "__main__":
    cli.run_app(server)
