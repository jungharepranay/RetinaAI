"""
llm_explainer.py
----------------
LLM-based explanation module for RetinAI.

Generates natural-language explanations of clinical screening results
using LLM APIs (Google Gemini primary, Groq secondary). Falls back to
template-based explanations when all API calls fail.

SAFETY:
- The LLM is used ONLY for explanation - NEVER for scoring or prediction.
- The LLM receives structured clinical assessment, not raw model output.
- System prompt explicitly forbids adding/removing diagnoses.
- Temperature is set to 0 for deterministic output.
- All LLM output carries an "AI-generated explanation" label.

Usage:
    Set GEMINI_API_KEY and/or GROQ_API_KEY in environment or .env file.
    Fallback chain: Gemini -> Groq -> Template.
"""

import os
import json
import sys

# ================================================================== #
#  ENVIRONMENT SETUP                                                   #
# ================================================================== #

def _log(msg: str) -> None:
    """Print to stderr with UTF-8 encoding to avoid cp1252 issues on Windows."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        # Fallback: write to stderr with utf-8
        sys.stderr.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
        sys.stderr.flush()


_log("[llm_explainer] >> Loading Gemini API key...")

# Load .env file if present (so user can just edit .env)
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), ".env")
    load_dotenv(_env_path)
    _log(f"[llm_explainer] >> Loaded .env from: {_env_path}")
except ImportError:
    _log("[llm_explainer] !! python-dotenv not installed -- using env vars directly")

# ================================================================== #
#  LLM CONFIGURATION                                                   #
# ================================================================== #

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"

_log(f"[llm_explainer] >> API KEY FOUND: {bool(GEMINI_API_KEY)}")
if GEMINI_API_KEY:
    _log(f"[llm_explainer] >> Key prefix: {GEMINI_API_KEY[:10]}...")
else:
    _log("[llm_explainer] !! ERROR: GEMINI_API_KEY not found in environment")

# Detect which Gemini SDK is available
_GENAI_CLIENT = None
_SDK_TYPE = None  # "new" for google-genai, "legacy" for google-generativeai

try:
    from google import genai
    _GENAI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    _SDK_TYPE = "new"
    _log(f"[llm_explainer] >> Using google-genai SDK (new), version: {genai.__version__}")
except ImportError:
    try:
        import google.generativeai as genai_legacy
        genai_legacy.configure(api_key=GEMINI_API_KEY)
        _SDK_TYPE = "legacy"
        _log("[llm_explainer] >> Using google-generativeai SDK (legacy)")
    except ImportError:
        _SDK_TYPE = None
        _log("[llm_explainer] !! No Gemini SDK installed -- will use REST API fallback")
except Exception as e:
    _log(f"[llm_explainer] !! SDK init error: {e}")

# ------------------------------------------------------------------ #
#  GROQ SETUP                                                          #
# ------------------------------------------------------------------ #

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
_GROQ_CLIENT = None

if GROQ_API_KEY:
    try:
        from groq import Groq
        _GROQ_CLIENT = Groq(api_key=GROQ_API_KEY)
        _log(f"[llm_explainer] >> Groq client initialized (model: {GROQ_MODEL})")
    except ImportError:
        _log("[llm_explainer] !! groq package not installed -- Groq fallback unavailable")
    except Exception as e:
        _log(f"[llm_explainer] !! Groq init error: {e}")
else:
    _log("[llm_explainer] !! GROQ_API_KEY not found -- Groq fallback unavailable")


SYSTEM_PROMPT = """You are a clinical assistant helping explain retinal screening results.

You MUST strictly follow these rules:

STRUCTURE:
You MUST output using EXACTLY these sections:

Summary:
(2-3 short sentences only)

Key Findings:
* Use bullet points
* Include ONLY findings provided in input
* Format: <Condition> — <Risk Level> (<confidence %>)

What This Means:
* Explain implications in simple terms
* 2-3 sentences maximum

Next Steps:
* Provide clear, actionable advice
* 2-4 bullet points

SAFETY RULES:
* DO NOT introduce new diseases
* DO NOT remove or change predicted diseases
* DO NOT reinterpret or override confidence values
* DO NOT make a medical diagnosis
* Use phrases like 'suggests risk', 'indicates possibility'

RISK HANDLING:
* Use the provided risk labels (High / Borderline / Low)
* DO NOT infer risk from confidence numbers
* If confidence is high but risk is borderline, follow the given risk label

STYLE:
* Be concise and clear
* Avoid repetition
* Avoid long paragraphs
* Use simple, patient-friendly language
"""

QA_SYSTEM_PROMPT = """You are a clinical assistant helping patients understand their retinal screening results.

Your task is to DIRECTLY ANSWER the patient's specific question.

RULES:
* Read the patient's question carefully and answer ONLY what they are asking.
* Do NOT produce a full clinical summary or structured report.
* Do NOT output sections like "Summary:", "Key Findings:", "What This Means:", or "Next Steps:" unless the question specifically asks for one of those.
* Keep your answer focused, concise, and conversational (3-8 sentences).
* Use simple, patient-friendly language.
* Base your answer ONLY on the provided clinical assessment data.

SAFETY RULES:
* DO NOT introduce new diseases or conditions not present in the data.
* DO NOT remove or change predicted diseases.
* DO NOT reinterpret or override confidence values.
* DO NOT make a medical diagnosis.
* Use phrases like 'suggests risk', 'indicates possibility'.
* Use the provided risk labels (High / Borderline / Low) exactly as given.
"""


# ================================================================== #
#  GEMINI API CALLS                                                    #
# ================================================================== #

def _call_gemini(prompt: str, system_prompt: str, max_tokens: int = 500) -> str:
    """
    Call Gemini API using whichever SDK is available.

    Tries: new SDK -> legacy SDK -> REST API.
    Returns empty string on failure (caller handles fallback).
    """
    if not GEMINI_API_KEY:
        _log("[llm_explainer] !! Cannot call Gemini: no API key")
        return ""

    _log(f"[llm_explainer] >> Sending request to Gemini ({GEMINI_MODEL})...")
    _log(f"[llm_explainer] >> Prompt length: {len(prompt)} chars")

    # --- Try new google-genai SDK (v1.x) --- #
    if _SDK_TYPE == "new" and _GENAI_CLIENT is not None:
        try:
            from google.genai import types

            response = _GENAI_CLIENT.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0,
                    max_output_tokens=max_tokens,
                ),
            )
            response_text = response.text or ""
            _log(f"[llm_explainer] OK Raw LLM response ({len(response_text)} chars): "
                 f"{response_text[:200]}...")
            return response_text

        except Exception as e:
            _log(f"[llm_explainer] !! google-genai SDK error: {e}")
            import traceback
            traceback.print_exc()
            # Fall through to REST

    # --- Try legacy google-generativeai SDK --- #
    if _SDK_TYPE == "legacy":
        try:
            import google.generativeai as genai_legacy
            model = genai_legacy.GenerativeModel(
                GEMINI_MODEL,
                system_instruction=system_prompt,
            )
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0,
                    "max_output_tokens": max_tokens,
                },
            )
            response_text = response.text or ""
            _log(f"[llm_explainer] OK Raw LLM response ({len(response_text)} chars): "
                 f"{response_text[:200]}...")
            return response_text

        except Exception as e:
            _log(f"[llm_explainer] !! google-generativeai SDK error: {e}")
            import traceback
            traceback.print_exc()
            # Fall through to REST

    # --- Fallback: REST API --- #
    return _call_gemini_rest(prompt, system_prompt, max_tokens)


def _call_gemini_rest(prompt: str, system_prompt: str, max_tokens: int = 500) -> str:
    """Last resort: call Gemini via REST API (no SDK needed)."""
    import urllib.request
    import urllib.error

    _log("[llm_explainer] >> Falling back to REST API...")

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}")

    payload = {
        "contents": [{
            "parts": [{"text": system_prompt + "\n\n" + prompt}]
        }],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": max_tokens,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            candidates = result.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    response_text = parts[0].get("text", "")
                    _log(f"[llm_explainer] OK REST response ({len(response_text)} chars): "
                         f"{response_text[:200]}...")
                    return response_text
            _log(f"[llm_explainer] !! REST returned no candidates: "
                 f"{json.dumps(result)[:300]}")
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        _log(f"[llm_explainer] !! Gemini REST error: {e}")
        if hasattr(e, 'read'):
            try:
                error_body = e.read().decode()
                _log(f"[llm_explainer] !! Error body: {error_body[:500]}")
            except Exception:
                pass
    except Exception as e:
        _log(f"[llm_explainer] !! Unexpected REST error: {e}")

    return ""


# ================================================================== #
#  GROQ API CALL                                                       #
# ================================================================== #

def _call_groq(prompt: str, system_prompt: str, max_tokens: int = 500) -> str:
    """
    Call Groq API as a fallback when Gemini is unavailable.

    Uses the same system prompt and temperature=0 for clinical safety.
    Returns empty string on failure.
    """
    if not _GROQ_CLIENT:
        _log("[llm_explainer] !! Cannot call Groq: no client available")
        return ""

    _log(f"[llm_explainer] >> Sending request to Groq ({GROQ_MODEL})...")

    try:
        completion = _GROQ_CLIENT.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )

        response_text = completion.choices[0].message.content or ""
        _log(f"[llm_explainer] OK Groq response ({len(response_text)} chars): "
             f"{response_text[:200]}...")
        return response_text

    except Exception as e:
        _log(f"[llm_explainer] !! Groq error: {e}")
        return ""


# ================================================================== #
#  PUBLIC API                                                          #
# ================================================================== #

def generate_llm_explanation(clinical_assessment: dict) -> dict:
    """
    Generate a natural-language explanation of screening results.

    Fallback chain: Gemini -> Groq -> Template.

    Parameters
    ----------
    clinical_assessment : dict
        Output of ``clinical_reasoning()`` function.

    Returns
    -------
    dict:
        llm_explanation : str  -- LLM-generated text (or template fallback)
        is_llm_generated : bool -- True if LLM was used
        provider : str -- "gemini" | "groq" | "template_fallback"
    """
    template_explanation = clinical_assessment.get("explanation", "")
    prompt = _build_prompt(clinical_assessment)

    # --- 1. Try Gemini --- #
    if GEMINI_API_KEY:
        llm_text = _call_gemini(prompt, SYSTEM_PROMPT, max_tokens=600)
        if llm_text and _validate_response_structure(llm_text):
            _log(f"[llm_explainer] OK Gemini explanation generated ({len(llm_text)} chars)")
            llm_text = _validate_llm_output(llm_text, clinical_assessment)
            return {
                "llm_explanation": llm_text,
                "is_llm_generated": True,
                "provider": "gemini",
            }
        _log("[llm_explainer] !! Gemini failed or response invalid -- switching to Groq")
    else:
        _log("[llm_explainer] !! No Gemini key -- trying Groq")

    # --- 2. Try Groq --- #
    if _GROQ_CLIENT:
        llm_text = _call_groq(prompt, SYSTEM_PROMPT, max_tokens=600)
        if llm_text and _validate_response_structure(llm_text):
            _log(f"[llm_explainer] OK Groq explanation generated ({len(llm_text)} chars)")
            llm_text = _validate_llm_output(llm_text, clinical_assessment)
            return {
                "llm_explanation": llm_text,
                "is_llm_generated": True,
                "provider": "groq",
            }
        _log("[llm_explainer] !! Groq failed or response invalid -- using template")
    else:
        _log("[llm_explainer] !! No Groq client -- using template")

    # --- 3. Template fallback --- #
    _log("[llm_explainer] >> Returning template fallback")
    return {
        "llm_explanation": template_explanation,
        "is_llm_generated": False,
        "provider": "template_fallback",
    }


def generate_qa_explanation(
    clinical_assessment: dict,
    user_question: str,
) -> dict:
    """
    Answer a user question about screening results using LLM.

    Fallback chain: Gemini -> Groq -> Template.

    Parameters
    ----------
    clinical_assessment : dict
        Output of ``clinical_reasoning()`` function.
    user_question : str
        The user's question about the results.

    Returns
    -------
    dict:
        answer : str
        is_llm_generated : bool
        provider : str -- "gemini" | "groq" | "template_fallback"
    """
    _log(f"[llm_explainer] >> QA request -- question: {user_question[:100]}")

    template_answer = _build_template_qa_answer(
        clinical_assessment, user_question
    )
    prompt = _build_qa_prompt(clinical_assessment, user_question)

    # --- 1. Try Gemini --- #
    if GEMINI_API_KEY:
        try:
            llm_text = _call_gemini(prompt, QA_SYSTEM_PROMPT, max_tokens=400)
        except Exception as e:
            _log(f"[llm_explainer] !! Gemini QA call error: {e}")
            llm_text = ""

        if llm_text and len(llm_text.strip()) > 30:
            _log(f"[llm_explainer] OK Gemini QA answer ({len(llm_text)} chars)")
            llm_text = _validate_llm_output(llm_text, clinical_assessment)
            return {
                "answer": llm_text,
                "is_llm_generated": True,
                "provider": "gemini",
            }
        _log("[llm_explainer] !! Gemini QA failed -- switching to Groq")
    else:
        _log("[llm_explainer] !! No Gemini key -- trying Groq for QA")

    # --- 2. Try Groq --- #
    if _GROQ_CLIENT:
        try:
            llm_text = _call_groq(prompt, QA_SYSTEM_PROMPT, max_tokens=400)
        except Exception as e:
            _log(f"[llm_explainer] !! Groq QA call error: {e}")
            llm_text = ""

        if llm_text and len(llm_text.strip()) > 30:
            _log(f"[llm_explainer] OK Groq QA answer ({len(llm_text)} chars)")
            llm_text = _validate_llm_output(llm_text, clinical_assessment)
            return {
                "answer": llm_text,
                "is_llm_generated": True,
                "provider": "groq",
            }
        _log("[llm_explainer] !! Groq QA failed -- using template")
    else:
        _log("[llm_explainer] !! No Groq client -- using template for QA")

    # --- 3. Template fallback --- #
    _log("[llm_explainer] >> Returning template QA fallback")
    return {
        "answer": template_answer,
        "is_llm_generated": False,
        "provider": "template_fallback",
    }


# ================================================================== #
#  PROMPT BUILDERS                                                     #
# ================================================================== #

def _build_prompt(assessment: dict) -> str:
    """Build a structured prompt for the LLM from clinical assessment.

    Sends structured finding objects so the LLM receives explicit
    condition / risk / confidence data rather than raw probabilities.
    """
    import json as _json

    lines = ["Please explain these retinal screening results:\n"]

    # Patient summary
    patient = assessment.get("patient_summary", "")
    if patient:
        lines.append(f"Patient: {patient}")

    # Key findings — structured objects
    key_findings = assessment.get("key_findings", [])
    if key_findings:
        lines.append("\nStructured findings (use EXACTLY these values):")
        structured_findings = []
        for f in key_findings:
            finding_obj = {
                "condition": f["disease"],
                "risk": f["risk_level"],
                "confidence": round(f["probability"], 4),
            }
            structured_findings.append(finding_obj)

            # Also provide human-readable line
            lines.append(f"- {f['disease']}: {f['risk_level']} "
                         f"(confidence: {f['probability']:.1%})")
            if f.get("description"):
                lines.append(f"  Description: {f['description']}")
            if f.get("supporting_evidence"):
                lines.append(f"  Evidence: {'; '.join(f['supporting_evidence'][:2])}")
            if f.get("contradicting_evidence"):
                lines.append(f"  Caution: {'; '.join(f['contradicting_evidence'][:2])}")

        lines.append(f"\nFindings JSON:\n{_json.dumps(structured_findings, indent=2)}")
    else:
        lines.append("\nNo significant findings detected.")

    # Urgency
    urgency = assessment.get("urgency", "routine")
    lines.append(f"\nOverall urgency: {urgency}")

    # Uncertainty
    if assessment.get("uncertain"):
        lines.append("\nNote: The model shows uncertainty in these results.")

    lines.append("\nIMPORTANT: Use the risk labels above (High Risk / Borderline / Low Risk) "
                 "exactly as given. Do NOT infer risk from confidence percentages.")
    lines.append("\nPlease provide a patient-friendly explanation using the "
                 "required sections: Summary, Key Findings, What This Means, Next Steps.")
    return "\n".join(lines)


def _build_qa_prompt(assessment: dict, question: str) -> str:
    """Build a Q&A prompt combining clinical data and user question.

    Unlike _build_prompt(), this does NOT include instructions to produce
    a structured summary — those instructions would override the user's
    question and cause the LLM to emit the same clinical summary every time.
    """
    import json as _json

    lines = ["CLINICAL ASSESSMENT DATA:\n"]

    # Patient summary
    patient = assessment.get("patient_summary", "")
    if patient:
        lines.append(f"Patient: {patient}")

    # Key findings
    key_findings = assessment.get("key_findings", [])
    if key_findings:
        lines.append("\nDetected findings:")
        for f in key_findings:
            lines.append(f"- {f['disease']}: {f['risk_level']} "
                         f"(confidence: {f['probability']:.1%})")
            if f.get("description"):
                lines.append(f"  Description: {f['description']}")
            if f.get("recommendation"):
                lines.append(f"  Recommendation: {f['recommendation']}")
    else:
        lines.append("\nNo significant findings detected.")

    # Urgency
    urgency = assessment.get("urgency", "routine")
    lines.append(f"\nOverall urgency: {urgency}")

    # Uncertainty
    if assessment.get("uncertain"):
        lines.append("Note: The model shows uncertainty in these results.")

    # The actual question
    lines.append(f"\n---\nUSER QUESTION:\n{question}")
    lines.append("\nAnswer the question above directly and concisely. "
                 "Do NOT produce a structured clinical summary.")

    return "\n".join(lines)


# ================================================================== #
#  VALIDATION                                                          #
# ================================================================== #

def _validate_response_structure(llm_text: str) -> bool:
    """
    Light validation: check that the LLM response has required sections.

    Accept response if:
    - Contains "Summary:"
    - Contains "Key Findings:"
    - Length > 50 characters

    Returns True if response is valid, False if fallback is needed.
    """
    if not llm_text or len(llm_text.strip()) < 50:
        _log("[llm_explainer] !! Response too short or empty")
        return False

    text_lower = llm_text.lower()
    has_summary = "summary:" in text_lower or "**summary:**" in text_lower or "**summary**" in text_lower
    has_findings = "key findings:" in text_lower or "**key findings:**" in text_lower or "**key findings**" in text_lower

    if not has_summary:
        _log("[llm_explainer] !! Response missing 'Summary:' section")
        return False
    if not has_findings:
        _log("[llm_explainer] !! Response missing 'Key Findings:' section")
        return False

    _log("[llm_explainer] OK Response structure validation passed")
    return True


def _validate_llm_output(llm_text: str, assessment: dict) -> str:
    """
    Validate LLM output for hallucinations.

    Checks for hallucinated conditions not in key findings.
    Does NOT reject the response -- only logs warnings.
    """
    # List of diseases that WERE found significant
    found_diseases = {f["disease"].lower()
                      for f in assessment.get("key_findings", [])}

    # Expand with related terms (so "diabetes" isn't flagged when
    # "diabetic retinopathy" is a finding, etc.)
    _related_terms = {
        "diabetic retinopathy": {"diabetes", "diabetic"},
        "age-related macular degeneration": {"amd", "macular degeneration"},
        "amd": {"age-related macular degeneration", "macular degeneration"},
        "macular degeneration": {"amd", "age-related macular degeneration"},
    }
    expanded = set(found_diseases)
    for fd in found_diseases:
        if fd in _related_terms:
            expanded.update(_related_terms[fd])

    # List of all known diseases to check against
    all_diseases = {
        "diabetic retinopathy", "diabetes", "glaucoma", "cataract",
        "age-related macular degeneration", "amd", "hypertension",
        "myopia", "macular degeneration",
    }

    # Check if LLM mentioned diseases not in findings
    text_lower = llm_text.lower()
    hallucinated = []
    for disease in all_diseases:
        if disease in text_lower:
            # Check if it's in our findings (fuzzy match)
            if not any(disease in fd or fd in disease
                       for fd in expanded):
                # Allow if it's mentioned in a "not found" context
                if f"no {disease}" not in text_lower and \
                   f"not {disease}" not in text_lower and \
                   f"no signs of {disease}" not in text_lower and \
                   f"no evidence of {disease}" not in text_lower and \
                   f"without {disease}" not in text_lower and \
                   f"absence of {disease}" not in text_lower:
                    hallucinated.append(disease)

    if hallucinated:
        _log(f"[llm_explainer] !! Hallucination guard triggered for: {hallucinated}")
    else:
        _log("[llm_explainer] OK Hallucination check passed")

    return llm_text


# ================================================================== #
#  TEMPLATE FALLBACK                                                   #
# ================================================================== #

def _build_template_qa_answer(
    assessment: dict, question: str
) -> str:
    """
    Generate a safe template-based answer when the LLM is unavailable.
    """
    findings = assessment.get("key_findings", [])
    urgency = assessment.get("urgency", "routine")
    q_lower = question.lower()

    # Determine answer based on common question patterns
    if any(w in q_lower for w in ["mean", "result", "finding"]):
        if not findings:
            body = ("Your screening results did not identify any "
                    "significant retinal abnormalities. This is "
                    "a positive outcome, but routine follow-up eye "
                    "exams are still recommended.")
        else:
            diseases = ", ".join(f["disease"] for f in findings)
            body = (f"The screening identified potential indicators "
                    f"for: {diseases}. These findings should be "
                    f"evaluated by an ophthalmologist for confirmation.")

    elif any(w in q_lower for w in ["serious", "severity", "bad", "worry"]):
        if urgency == "urgent":
            body = ("The screening detected findings that may need "
                    "prompt clinical attention. We recommend "
                    "scheduling a consultation as soon as possible.")
        elif urgency == "follow-up" or findings:
            body = ("Some findings warrant clinical follow-up. While "
                    "this screening is not a diagnosis, we recommend "
                    "discussing these results with a specialist.")
        else:
            body = ("No significant concerns were flagged. Continue "
                    "with routine eye care and regular check-ups.")

    elif any(w in q_lower for w in ["next", "do", "action", "step"]):
        if findings:
            body = ("We recommend consulting an ophthalmologist to "
                    "discuss these screening results. Bring this "
                    "report to your appointment for reference.")
        else:
            body = ("Continue with routine eye examinations as "
                    "recommended for your age group. No immediate "
                    "action is needed based on this screening.")

    elif any(w in q_lower for w in ["doctor", "consult", "specialist"]):
        if findings or urgency != "routine":
            body = ("Yes, we recommend consulting an eye care "
                    "specialist (ophthalmologist) to discuss these "
                    "screening findings. They can perform a "
                    "comprehensive evaluation.")
        else:
            body = ("While no significant findings were detected, "
                    "regular eye check-ups are always beneficial, "
                    "especially as you age.")

    else:
        # Generic response
        body = ("Based on your screening results, we recommend "
                "consulting a qualified healthcare professional "
                "who can provide personalized guidance about your "
                "eye health.")

    return body
