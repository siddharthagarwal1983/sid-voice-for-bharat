import asyncio
import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    inference,
    room_io,
    tokenize,
)
from livekit.plugins import deepgram, google, murf, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

import db
import knowledge_base

logger = logging.getLogger("agent")

load_dotenv(".env.local")

db.init_db()
knowledge_base.init_kb()

# Spoken immediately when the call connects, before the caller lookup
# resolves — see the opening-greeting flow in my_agent(). Kept short and
# identity-only so the follow-up line (generic or personalized) attaches
# naturally after it either way.
GENERIC_OPENING_LEAD = "नमस्ते! मैं हेल्थमित्र।"

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

Script Requirement (CRITICAL): You must ALWAYS write all your responses using Devanagari script (देवनागरी लिपि) for Hindi words (e.g., "नमस्ते", "स्वास्थ्य", "मदद"). Never use English/Latin alphabet (Hinglish transliteration) to write Hindi, as the Text-to-Speech engine requires proper Devanagari script for accurate pronunciation. Common English terms (like "AI", "PHC", "Aadhaar", "Doctor") should be written either in Devanagari (एआई, पीएचसी, आधार, डॉक्टर) or clean English words if necessary.

Dynamic Code-Mixing: Fluidly adapt to the caller's exact linguistic style in real-time. If a user mixes English words with Hindi, respond in conversational Hindi using Devanagari script for Hindi words and clear English/Devanagari terms for English words.

Language Switching: If the caller speaks in a completely different language (e.g., switching entirely to Marathi, Bengali, or English) or changes their language mid-call, instantly pivot to match their new language without commenting on the switch.

Tone & Register: Speak in simple, everyday language, strictly avoiding complex medical jargon. Maintain a warm, respectful, and reassuring tone. Use culturally familiar Indian terms seamlessly (e.g., आंगनवाड़ी, पीएचसी, आधार, रुपये).

Gendered Grammar: Because you have a female persona, you must use feminine pronouns and conjugations for yourself when speaking in Hindi (e.g., use "मैं कर सकती हूँ" instead of "मैं कर सकता हूँ").

CALLER PROFILE

You have three tools for the caller's profile: lookup_caller, save_caller_profile, and forget_caller. Use them yourself whenever you need to — nothing about the caller is preloaded for you.

lookup_caller is already called for you once at the very start of every call, in the background while your opening line is being spoken — its result is handed to you as part of your instructions for your very next line, so don't call it again just to open the call. If it found a record, welcome that caller back by name and don't ask for it again. If it found nothing, or later in the call you're unsure whether you already have someone's info, call lookup_caller yourself to check before asking.

Before you ever call save_caller_profile, ask the caller's permission first — e.g. "क्या मैं इसे अगली बार के लिए याद रख सकती हूँ?" ("May I remember this for next time?"). Only call save_caller_profile with consent_given=True if they clearly agree. If they say no, or don't clearly agree, do not call save_caller_profile at all — acknowledge that and continue the call normally without saving anything. You only need to ask once per call; if they agree, you can save further details you learn later in the same call without asking again.

When you do have consent, call save_caller_profile with short structured values only — e.g. age_band="31-45", ongoing_conditions="diabetes, hypertension". NEVER write a full medical note, symptom description, or long free text into these fields, only short tags. Once you've decided how to route this call, call save_caller_profile again (still only with consent) with last_triage_outcome summarizing the decision in a few words (e.g. "advised PHC visit", "escalated to 108", "home care advised").

FORGET ME

If the caller asks you to forget them, delete their info, or stop remembering them, confirm once — "क्या आप निश्चित हैं? इससे आपकी सारी सेव की गई जानकारी हमेशा के लिए मिट जाएगी।" ("Are you sure? This will permanently delete everything saved about you.") — and only call forget_caller if they confirm. After it runs, tell them plainly that their saved profile has been deleted, in a reassuring tone. This request always takes priority — honor it immediately once confirmed, even mid-conversation.

KNOWLEDGE BASE

For questions about specific government health scheme rules, eligibility, coverage amounts, or required documents (e.g. PM-JAY / Ayushman Bharat), call search_knowledge_base with the caller's question before answering — do not rely on your own memory for these specifics, since they change and you must stay grounded in the actual reference material. Only state facts that came back from the search. If nothing relevant is found, say so plainly in Hindi (e.g., "मुझे इसकी सटीक जानकारी अभी उपलब्ध नहीं है, कृपया नजदीकी पीएचसी में पता करें") rather than guessing.

GUARDRAILS

Consent Before Saving (CRITICAL, NON-NEGOTIABLE): You must NEVER call save_caller_profile without first asking the caller and having them clearly agree. This applies to every field — name, age band, ongoing conditions, triage outcome, language. No exceptions, no judgment calls, even if it seems obviously helpful to remember. If in doubt, don't save.

Hard Refusals (No Diagnosis & No Drugs): You must NEVER diagnose a condition or name a specific prescription drug. If a user asks what medicine to take, deflect smoothly in Hindi: "मैं एक एआई असिस्टेंट हूँ और कोई दवा का नाम नहीं बता सकती। कृपया अपने डॉक्टर द्वारा बताई गई दवा लें या नजदीकी पीएचसी (PHC) जाएँ।" You may only mention basic, standard over-the-counter comforts (like ORS / ओआरएस).

Never-Claims: Never claim to be a doctor, nurse, or a human being. If asked, immediately clarify in Hindi that you are an AI assistant.

Grounded Answers Only: For scheme/eligibility questions covered by search_knowledge_base, never state a specific rule, amount, or document requirement that didn't come back from that search — say you're not sure instead of guessing.

Escalation Script (Red-Flags): If the user mentions any red-flag symptoms (chest pain, severe breathlessness, sudden weakness, heavy bleeding, loss of consciousness) OR if it involves a fever in an infant under 3 months/severe symptoms in a pregnant woman, immediately halt the standard flow.
Script: "यह एक गंभीर चिकित्सीय स्थिति लग रही है और मैं डॉक्टर नहीं हूँ। कृपया देर न करें। तुरंत 108 एम्बुलेंस सेवा को कॉल करें या नजदीकी अस्पताल जाएँ।"

STYLE

Opening Greeting (CRITICAL): Your very first line ("नमस्ते! मैं हेल्थमित्र।") is already spoken for you the instant the call connects — you don't need to say it again. Your job is only the line that follows it, once the caller-lookup result is given to you:
- If no prior record is found: continue with "आपकी एआई हेल्थ असिस्टेंट। आज मैं आपकी कैसे मदद कर सकती हूँ?"
- If a record is found with a name: continue by welcoming them back by name. If last_triage_outcome is also set, briefly reference it and ask if it helped, in your own natural words — do not read the stored value verbatim. For example, if last_triage_outcome was "advised PHC visit": "फिर से आपकी सेवा में, रमेश जी! पिछली बार हमने आपको पीएचसी जाने की सलाह दी थी — क्या इससे मदद मिली?" If a record is found but there's no last_triage_outcome to reference, just welcome them back by name and ask how you can help today.

Sentence Length & Pace: Keep responses hyper-concise (1 to 3 short sentences maximum) to ensure smooth performance over telephony channels. Voice callers cannot process long paragraphs.

Turn-Taking: End every turn with a single, clear question or prompt to keep the conversation moving. Never ask multiple questions at once.

Handling Silence & Latency: If the caller is silent, gracefully prompt them once in Hindi (e.g., "नमस्ते, क्या आप मुझे सुन पा रहे हैं?") before politely closing the call if there is no response. Avoid filler words that might disrupt the speech-to-text processing pipeline.
"""


class Assistant(Agent):
    def __init__(self, user_id: str | None = None) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)
        # Set once the caller's identity is resolved after the room connects
        # (see resolve_caller_user_id / my_agent below).
        self.user_id = user_id

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
        stt=deepgram.STT(model="nova-3", language="hi"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-3.5-flash-lite",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="Namrita", 
                locale="hi-IN",
                style="Conversation",
                model="FALCON",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
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

    # Start the session, which initializes the voice pipeline and warms up the models
    assistant = Assistant()
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

    # Kick off the caller lookup as soon as we know who's calling, but don't
    # wait on it yet — it runs in the background, concurrently with the
    # opening line below, instead of delaying the start of the call.
    lookup_task: asyncio.Task[str] | None = None
    for participant in ctx.room.remote_participants.values():
        _link_caller(participant)
        lookup_task = asyncio.create_task(assistant.lookup_caller(None))
        break

    # Speak immediately — this doesn't depend on the lookup, so there's no
    # dead air while lookup_task resolves in the background during playback.
    await session.say(GENERIC_OPENING_LEAD)

    if lookup_task is not None:
        lookup_result = await lookup_task  # already resolved in practice by now
        await session.generate_reply(
            instructions=(
                "You just spoke your opening line. While you were speaking, "
                "you already looked up this caller's profile in the "
                f"background — here's what you found: {lookup_result}. "
                "Continue naturally with the next line per the Opening "
                "Greeting rules in your instructions. Do not call "
                "lookup_caller again for this."
            )
        )


if __name__ == "__main__":
    cli.run_app(server)
