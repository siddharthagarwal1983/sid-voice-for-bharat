"""The two domain lookups HealthMitra can't do its job without.

1. classify_triage — routes a caller's described symptoms to a triage level
   (emergency / phc / home_care). This is a hand-built, deterministic
   keyword ruleset modeled on the same red-flag list already in the system
   prompt (standard ASHA/PHC referral guidance) — NOT a live medical API.
   There isn't a reliable public API for this, and for a no-diagnosis
   assistant a fixed, inspectable ruleset is arguably the *right* choice
   anyway: every routing decision traces back to a specific keyword, never
   to an LLM guess. See README for the "local, not live" disclosure.

2. find_facilities — looks up nearby government health facilities for a
   district. Tries a LIVE fetch against OpenStreetMap's free, keyless
   Nominatim search API first. If that fails (timeout, rate limit, no
   network — all realistic on the mobile connections these callers use),
   it falls back to a small hand-curated LOCAL list of well-known
   government hospitals for a handful of major districts, and reports
   which source it actually used. See README for current live-path status.
"""

import logging
import threading
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger("agent.health_tools")

# ---------------------------------------------------------------------------
# 1. Symptom -> triage level (local, rule-based — see module docstring)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriageRule:
    level: str  # "emergency" | "phc" | "home_care"
    keywords: tuple[str, ...]
    advice_en: str
    advice_hi: str


# Order matters: checked top to bottom, first match wins. Emergency keywords
# must be checked before the milder buckets so e.g. "बुखार और सांस लेने में
# तकलीफ" (fever + breathlessness) routes to emergency, not home_care.
_TRIAGE_RULES: tuple[TriageRule, ...] = (
    TriageRule(
        level="emergency",
        keywords=(
            "chest pain",
            "सीने में दर्द",
            "छाती में दर्द",
            "breathless",
            "साँस लेने में तकलीफ",
            "सांस लेने में तकलीफ",
            "सांस फूलना",
            "सांस नहीं आ रही",
            "unconscious",
            "बेहोश",
            "होश नहीं",
            "heavy bleeding",
            "बहुत खून बह",
            "तेज़ खून बह",
            "sudden weakness",
            "अचानक कमजोरी",
            "लकवा",
            "paralysis",
            "seizure",
            "मिर्गी का दौरा",
            "झटके आना",
            "असहनीय दर्द",
        ),
        advice_en=(
            "This looks like a serious medical situation. Please don't delay — "
            "call the 108 ambulance service right now, or go to the nearest hospital."
        ),
        advice_hi=(
            "यह एक गंभीर चिकित्सीय स्थिति लग रही है। कृपया देर न करें — "
            "तुरंत 108 एम्बुलेंस सेवा को कॉल करें या नजदीकी अस्पताल जाएँ।"
        ),
    ),
    TriageRule(
        level="phc",
        keywords=(
            "high fever",
            "तेज़ बुखार",
            "तेज बुखार",
            "बुखार तीन दिन",
            "लगातार बुखार",
            "vomiting",
            "उल्टी",
            "diarrhea",
            "दस्त",
            "dehydrat",
            "पेशाब कम",
            "wound",
            "घाव से खून",
            "deep cut",
            "गहरा घाव",
            "लगातार खांसी",
            "हफ्ते से खांसी",
            "pregnan",
            "गर्भवती",
            "infant",
            "नवजात",
            "shishu",
            "शिशु",
        ),
        advice_en=(
            "These symptoms don't look like something to manage at home. "
            "Please visit your nearest PHC today."
        ),
        advice_hi=(
            "यह लक्षण घर पर संभालने लायक नहीं लगते। कृपया आज ही नजदीकी पीएचसी (PHC) में दिखा लें।"
        ),
    ),
    TriageRule(
        level="home_care",
        keywords=(
            "mild fever",
            "हल्का बुखार",
            "cold",
            "जुकाम",
            "cough",
            "खांसी",
            "headache",
            "सरदर्द",
            "सिरदर्द",
            "body ache",
            "बदन दर्द",
            "minor cut",
            "छोटी चोट",
            "गले में खराश",
            "sore throat",
        ),
        advice_en=(
            "These look like mild symptoms. Rest, stay hydrated, and take ORS in "
            "small sips if needed. If it doesn't improve in 2-3 days or gets "
            "worse, please visit your nearest PHC."
        ),
        advice_hi=(
            "यह हल्के लक्षण लगते हैं। आराम करें, तरल पदार्थ लें, और जरूरत हो "
            "तो ORS थोड़ी-थोड़ी मात्रा में लें। अगर 2-3 दिन में सुधार न हो या "
            "हालत बिगड़े, तो नजदीकी पीएचसी में जरूर दिखाएं।"
        ),
    ),
)

RULESET_VERSION = "local-v1"


_UNCLEAR_ADVICE_EN = (
    "I can't clearly tell from these symptoms whether home care is fine or "
    "you should see a PHC. Could you tell me a bit more, or just visit your "
    "nearest PHC directly?"
)
_UNCLEAR_ADVICE_HI = (
    "मैं इन लक्षणों से साफ तौर पर तय नहीं कर पा रही कि घर पर संभालना "
    "ठीक रहेगा या पीएचसी जाना चाहिए। कृपया थोड़ा और बताएं, या सीधे "
    "नजदीकी पीएचसी में दिखा लें।"
)


def classify_triage(symptom_text: str, language: str = "en") -> dict:
    """Pure, deterministic, offline — no network call. Returns a dict, never
    raises, so it's always safe for a tool wrapper to call directly.

    `language` ("en" or "hi") picks which of the ruleset's two fixed advice
    strings comes back as `advice` — the caller (agent.py) passes whichever
    language it's currently speaking with the caller, so the UI card and the
    spoken line always match the conversation, never defaulting to Hindi
    regardless of what language the call is actually in.
    """
    is_hindi = language.lower().startswith("hi")
    text = (symptom_text or "").lower()
    for rule in _TRIAGE_RULES:
        for keyword in rule.keywords:
            if keyword.lower() in text:
                return {
                    "level": rule.level,
                    "matched_keyword": keyword,
                    "advice": rule.advice_hi if is_hindi else rule.advice_en,
                    "language": "hi" if is_hindi else "en",
                    "source": "local-ruleset",
                    "ruleset_version": RULESET_VERSION,
                }
    return {
        "level": "unclear",
        "matched_keyword": None,
        "advice": _UNCLEAR_ADVICE_HI if is_hindi else _UNCLEAR_ADVICE_EN,
        "language": "hi" if is_hindi else "en",
        "source": "local-ruleset",
        "ruleset_version": RULESET_VERSION,
    }


# ---------------------------------------------------------------------------
# 2. District -> nearby facility (live OSM Nominatim, local fallback)
# ---------------------------------------------------------------------------

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_USER_AGENT = "HealthMitraVoiceAgentDemo/1.0 (student project; no production traffic)"
_MIN_REQUEST_GAP_S = 1.1  # Nominatim usage policy allows ~1 request/second
_TIMEOUT_S = 6.0  # keep short — the caller is on a live phone call

_last_request_at = 0.0
_request_lock = threading.Lock()

# Hand-curated LOCAL fallback, used only when the live lookup fails or times
# out. Real, well-known government hospitals (not scraped from a live feed),
# keyed by lowercase district/city name. Areas are approximate and NOT
# guaranteed current — see README's "local, not live" disclosure.
_LOCAL_FALLBACK: dict[str, list[dict]] = {
    "pune": [
        {
            "name": "Sassoon General Hospital",
            "type": "Government Hospital",
            "area": "Pune",
        },
        {
            "name": "PMC Kamla Nehru Hospital",
            "type": "Municipal Hospital",
            "area": "Pune",
        },
    ],
    "mumbai": [
        {
            "name": "KEM Hospital",
            "type": "Government Hospital",
            "area": "Parel, Mumbai",
        },
        {
            "name": "Sion Hospital (LTMGH)",
            "type": "Government Hospital",
            "area": "Sion, Mumbai",
        },
    ],
    "delhi": [
        {
            "name": "AIIMS Delhi",
            "type": "Government Hospital",
            "area": "Ansari Nagar, Delhi",
        },
        {"name": "Lok Nayak Hospital", "type": "Government Hospital", "area": "Delhi"},
    ],
    "bengaluru urban": [
        {
            "name": "Victoria Hospital",
            "type": "Government Hospital",
            "area": "Bengaluru",
        },
    ],
    "bengaluru": [
        {
            "name": "Victoria Hospital",
            "type": "Government Hospital",
            "area": "Bengaluru",
        },
    ],
    "chennai": [
        {
            "name": "Rajiv Gandhi Govt. General Hospital",
            "type": "Government Hospital",
            "area": "Chennai",
        },
    ],
    "lucknow": [
        {
            "name": "King George's Medical University",
            "type": "Government Hospital",
            "area": "Lucknow",
        },
    ],
    "patna": [
        {
            "name": "Patna Medical College Hospital",
            "type": "Government Hospital",
            "area": "Patna",
        },
    ],
    "jaipur": [
        {
            "name": "Sawai Man Singh Hospital",
            "type": "Government Hospital",
            "area": "Jaipur",
        },
    ],
    "ahmedabad": [
        {
            "name": "Civil Hospital Ahmedabad",
            "type": "Government Hospital",
            "area": "Ahmedabad",
        },
    ],
    "nagpur": [
        {
            "name": "Government Medical College & Hospital",
            "type": "Government Hospital",
            "area": "Nagpur",
        },
    ],
}


def _rate_limited_get(params: dict) -> httpx.Response:
    """Nominatim's usage policy caps free use at ~1 request/second. A single
    module-level lock + timestamp is enough here since one agent process
    handles one call's tool calls at a time.
    """
    global _last_request_at
    with _request_lock:
        wait = _MIN_REQUEST_GAP_S - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        response = httpx.get(
            _NOMINATIM_URL,
            params=params,
            headers={"User-Agent": _USER_AGENT},
            timeout=_TIMEOUT_S,
        )
        _last_request_at = time.monotonic()
        return response


def _fetch_live(district: str) -> list[dict] | None:
    """Best-effort live fetch. Returns None on ANY failure — timeout, no
    network, rate limit, bad response — so the caller can fall back cleanly
    instead of an exception surfacing mid-call.
    """
    try:
        response = _rate_limited_get(
            {
                "q": f"hospital in {district}, India",
                "format": "jsonv2",
                "limit": 5,
                "addressdetails": 1,
            }
        )
        response.raise_for_status()
        results = response.json()
    except (httpx.TimeoutException, httpx.HTTPError, ValueError) as exc:
        logger.warning("Live facility lookup failed for %r: %s", district, exc)
        return None

    facilities = []
    for item in results:
        name = item.get("name") or (item.get("display_name") or "").split(",")[0]
        if not name:
            continue
        facilities.append(
            {
                "name": name,
                "type": (item.get("type") or "health facility").replace("_", " "),
                "area": item.get("display_name", ""),
            }
        )
    return facilities or None


def find_facilities(district: str) -> dict:
    """Returns what was found (or not) for `district`, always stating where
    the data came from and when it was fetched. Never raises — a network
    failure here must become a spoken fallback, not a dead call.
    """
    now = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

    if not district or not district.strip():
        return {
            "status": "no_district",
            "district": district or "",
            "facilities": [],
            "data_source": None,
            "fetched_at": now,
        }

    normalized = district.strip()

    live = _fetch_live(normalized)
    if live:
        return {
            "status": "ok",
            "district": normalized,
            "facilities": live[:5],
            "data_source": "live:openstreetmap-nominatim",
            "fetched_at": now,
        }

    local = _LOCAL_FALLBACK.get(normalized.lower())
    if local:
        return {
            "status": "ok",
            "district": normalized,
            "facilities": local,
            "data_source": "local-fallback",
            "fetched_at": now,
        }

    return {
        "status": "not_found",
        "district": normalized,
        "facilities": [],
        "data_source": None,
        "fetched_at": now,
    }
