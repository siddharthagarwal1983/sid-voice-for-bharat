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
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    room_io,
    tokenize,
)
from livekit.plugins import deepgram, google, murf, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

import db
import health_tools
import knowledge_base

logger = logging.getLogger("agent")

load_dotenv(".env.local")

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

OUTBOUND CALLS (proactive reminders & follow-ups)

Most calls are inbound — the caller reached you. Some calls are ones YOU are placing (see backend/scripts/make_outbound_call.py) for a specific reason. For these, your instructions for the very first thing you say will tell you the call_type and any detail, plus the caller profile lookup already run for you — don't call lookup_caller again for this. Outbound calls always open differently from inbound: the person didn't choose to call, so identify yourself and the reason for calling immediately, in 1-2 sentences, before anything else. Always in English first, per the Default Language rule.

- call_type="medication_reminder": Introduce yourself briefly as HealthMitra, an AI health assistant calling with a reminder, then state the reminder clearly using the given detail — e.g. "Hi, this is HealthMitra — just a reminder to take your evening metformin dose." Then ask if they've taken it, or if they have any questions, and continue naturally from there.

- call_type="vaccination_reminder": Same pattern, but for a vaccination due — state which vaccine/dose using the given detail, then ask if they have any questions or would like help finding a facility (offer find_nearby_health_facility if so).

- call_type="triage_followup": You're calling to check in after a PREVIOUS call where you routed this caller somewhere. Introduce yourself, then reference their last_triage_outcome from the caller profile lookup in your own natural words — never read it verbatim — and ask how it went or how they're feeling now. Example, if last_triage_outcome was "advised PHC visit": "Hi, this is HealthMitra — I'm calling to follow up on our last conversation. You'd mentioned going to the PHC — were you able to, and how are you feeling now?" If there's no last_triage_outcome on file, ask generally how they've been feeling since you last spoke instead of referencing anything specific.

Whichever type it is, once you've delivered the reason for calling and gotten a response, the rest of the call proceeds exactly like any other — same tools, same Default Language rule, same guardrails — and call end_call per the Ending the Call rule below once it's naturally finished.

Voicemail (CRITICAL): On an outbound call, what picks up isn't always a person. If what you hear after your opening line sounds like an answering machine or voicemail system — a scripted greeting inviting you to leave a message after a tone, mentioning a mailbox, saying the person is unavailable, or the Hindi equivalent (e.g. "कृपया संदेश छोड़ें", "अभी उपलब्ध नहीं है") — call report_voicemail_detected immediately. Then leave ONE short, appropriate message and hang up: for a medication/vaccination reminder, state the reminder briefly and say you'll try again later; for a triage follow-up, say you were checking in and you'll try again later. Do not try to have a conversation with a voicemail greeting, and do not wait through a long greeting before acting — as soon as you're confident it's not live, act. If you're not genuinely confident it's a machine, treat it as a person instead and continue normally — a real but slow or unusually-worded reply is not voicemail.

GUARDRAILS

Consent Before Saving (CRITICAL, NON-NEGOTIABLE): You must NEVER call save_caller_profile without first asking the caller and having them clearly agree. This applies to every field — name, age band, ongoing conditions, triage outcome, language. No exceptions, no judgment calls, even if it seems obviously helpful to remember. If in doubt, don't save.

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
"""


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
        this tool — never call it before you've said goodbye. This tool
        itself doesn't speak anything; it waits for your farewell to finish
        playing, then disconnects the call.
        """
        # Let the farewell that was just generated finish playing before
        # actually hanging up, so it isn't cut off mid-sentence. Must use
        # RunContext.wait_for_playout() here, not the SpeechHandle's own —
        # this tool call is itself part of that same speech handle, so
        # waiting on the handle directly would be a circular self-wait.
        await context.wait_for_playout()

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

        return "\n\n".join(f"[{r['source']}] {r['text']}" for r in results)

    async def _publish_to_room(self, topic: str, payload: dict) -> None:
        """Best-effort push of a structured tool result to the caller's
        screen (if a UI is attached to this room). Never lets a UI publish
        failure interrupt the voice call.
        """
        if self.room is None:
            return
        try:
            await self.room.local_participant.send_text(
                json.dumps(payload, ensure_ascii=False), topic=topic
            )
        except Exception:
            logger.warning("Failed to publish %s data to room", topic, exc_info=True)

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
    {"medication_reminder", "vaccination_reminder", "triage_followup"}
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

    @session.on("conversation_item_added")
    def _on_conversation_item_added(event):
        db.save_message(
            db_session_id,
            event.item.role,
            event.item.text_content,
            event.item.created_at,
        )

    @session.on("close")
    def _on_close(event):
        db.end_call_session(db_session_id, event.reason.value)
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
