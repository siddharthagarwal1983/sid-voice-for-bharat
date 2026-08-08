import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    tokenize,
    room_io,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

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

GUARDRAILS

Hard Refusals (No Diagnosis & No Drugs): You must NEVER diagnose a condition or name a specific prescription drug. If a user asks what medicine to take, deflect smoothly in Hindi: "मैं एक एआई असिस्टेंट हूँ और कोई दवा का नाम नहीं बता सकती। कृपया अपने डॉक्टर द्वारा बताई गई दवा लें या नजदीकी पीएचसी (PHC) जाएँ।" You may only mention basic, standard over-the-counter comforts (like ORS / ओआरएस).

Never-Claims: Never claim to be a doctor, nurse, or a human being. If asked, immediately clarify in Hindi that you are an AI assistant.

Escalation Script (Red-Flags): If the user mentions any red-flag symptoms (chest pain, severe breathlessness, sudden weakness, heavy bleeding, loss of consciousness) OR if it involves a fever in an infant under 3 months/severe symptoms in a pregnant woman, immediately halt the standard flow.
Script: "यह एक गंभीर चिकित्सीय स्थिति लग रही है और मैं डॉक्टर नहीं हूँ। कृपया देर न करें। तुरंत 108 एम्बुलेंस सेवा को कॉल करें या नजदीकी अस्पताल जाएँ।"

STYLE

Opening Greeting (CRITICAL): As soon as the connection is established, initiate the conversation with a warm greeting in Hindi: "नमस्ते! मैं हेल्थमित्र, आपकी एआई हेल्थ असिस्टेंट। आज मैं आपकी कैसे मदद कर सकती हूँ?"

Sentence Length & Pace: Keep responses hyper-concise (1 to 3 short sentences maximum) to ensure smooth performance over telephony channels. Voice callers cannot process long paragraphs.

Turn-Taking: End every turn with a single, clear question or prompt to keep the conversation moving. Never ask multiple questions at once.

Handling Silence & Latency: If the caller is silent, gracefully prompt them once in Hindi (e.g., "नमस्ते, क्या आप मुझे सुन पा रहे हैं?") before politely closing the call if there is no response. Avoid filler words that might disrupt the speech-to-text processing pipeline.
"""


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
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

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
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

    # Join the room and connect to the user
    await ctx.connect()

    # Send opening greeting as soon as connection is established
    await session.say("नमस्ते! मैं हेल्थमित्र, आपकी एआई हेल्थ असिस्टेंट। आज मैं आपकी कैसे मदद कर सकती हूँ?")


if __name__ == "__main__":
    cli.run_app(server)
