"""Conservative extraction from user text; model arguments cannot authorize writes."""
import re

from langchain_core.messages import HumanMessage

from .state import HelpdeskContext

SECRET_PATTERNS = (
    r"\bsk-[A-Za-z0-9_-]{8,}",
    r"(?i)\b(?:my\s+)?(?:password|api[ _-]?key|recovery[ _-]?code)\s*(?:is\s+|[:=]\s*)[\w!@#$%^&*()\-+.]{2,}",
)


def redact(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = re.sub(pattern, "[REDACTED SECRET]", text)
    return text


def missing_ticket_details(helpdesk: HelpdeskContext) -> list[str]:
    missing = []
    if helpdesk.get("category") not in ("wifi", "password"):
        missing.append("supported problem category")
    if len(helpdesk.get("problem_description", "").strip()) < 10:
        missing.append("meaningful problem description")
    if not helpdesk.get("steps_known"):
        missing.append("steps already tried, or an explicit statement of none")
    return missing


def derive_helpdesk(messages, previous: HelpdeskContext | None = None) -> HelpdeskContext:
    """Retain exact reported spans; unfamiliar phrasing requires clarification."""
    result: HelpdeskContext = {
        "category": "", "problem_description": "", "steps_tried": [], "steps_known": False,
        "ticket_requested": False, "ticket_condition": "unconditional", "ticket_id": None,
        "ticket_request_id": None, "secret_detected": False,
    }
    request_number = 0
    for message in messages:
        if not isinstance(message, HumanMessage) or not isinstance(message.content, str):
            continue
        text = message.content
        lower = text.casefold()
        if redact(text) != text or "[REDACTED SECRET]" in text:
            result["secret_detected"] = True
            continue
        wifi = "wifi" in lower or "wi-fi" in lower
        password = "password" in lower
        if wifi or password:
            category = "ambiguous" if wifi and password else "wifi" if wifi else "password"
            if result["category"] and result["category"] != category:
                result.update(problem_description="", steps_tried=[], steps_known=False,
                              ticket_requested=False, ticket_condition="unconditional", ticket_id=None,
                              ticket_request_id=None)
            result["category"] = category
        requests_ticket = re.search(r"\b(?:prepare|create|draft|open|make|save)\b.{0,80}\b(?:ticket|draft)\b", lower)
        if re.search(r"\b(?:should|can|could|would|do|did)\s+(?:i|we)\b", lower):
            requests_ticket = None  # A question about what the user should do is not ticket consent.
        continues_condition = result["ticket_requested"] and re.match(r"\s*(?:only|if|unless|provided|when)\b", lower)
        if requests_ticket or continues_condition:
            # Negative or conditional wording we cannot establish is never permission.
            was_requested = result["ticket_requested"]
            new_issue = requests_ticket and re.search(
                r"\b(?:new|another|different|separate|second)\s+(?:(?:wi-?fi|password)\s+)?(?:issue|problem)\b", lower)
            explicitly_new = requests_ticket and (new_issue or re.search(
                r"\b(?:new|another|additional|separate|second)\s+(?:support\s+)?(?:ticket|draft)\b", lower))
            if new_issue:
                result.update(problem_description="", steps_tried=[], steps_known=False)
            if requests_ticket and (not was_requested or explicitly_new):
                request_number += 1
                result["ticket_request_id"] = f"request-{request_number}"
                result["ticket_id"] = None
                if explicitly_new:
                    result["ticket_condition"] = "unconditional"
                    was_requested = False
            result["ticket_requested"] = True
            ticket_text = " ".join(s for s in re.split(r"(?<=[.!?])\s+", lower)
                                   if re.search(r"\b(ticket|draft)\b", s))
            condition = "unconditional"
            if re.search(r"\b(if|unless|only|provided|when)\b", lower) or re.search(r"\b(don't|not|never)\b", ticket_text):
                condition = "unclear"
                if re.search(r"\bif (?:it|the service|wi-?fi) is (?:operating|operational|working|up)\s*[,.;]", lower):
                    condition = "operational"
                if re.search(r"\b(unless|only|provided|when)\b", lower) or re.search(r"\b(not|don't|never)\b", ticket_text):
                    condition = "unclear"
            # Do not silently erase an earlier condition on a vague follow-up.
            if condition != "unconditional" or not was_requested or re.search(r"\b(unconditional|unconditionally|regardless)\b", lower):
                result["ticket_condition"] = condition
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            reports_issue = re.search(r"(?i)\b(cannot|can't|unable|issue|problem|fails?|failed|will not|does not|broken|unavailable|no success)\b", sentence)
            names_issue = re.search(r"(?i)\b(wi-?fi|password|I|my|we)\b", sentence)
            if (reports_issue and names_issue and not sentence.rstrip().endswith("?")
                    and not re.search(r"(?i)\b(?:if|prepare|create|draft|ticket)\b", sentence)):
                result["problem_description"] = sentence
            if (re.search(r"(?i)\b(?:I|we)\s+(?:already\s+)?(?:enabled|forgot|tried|reconnected|restarted|followed|checked|reset|opened|disabled)\b", sentence)
                    and not re.search(r"(?i)\b(?:if|hypothetically|would)\b", sentence)):
                if sentence not in result["steps_tried"]:
                    result["steps_tried"].append(sentence)
                result["steps_known"] = True
            if re.search(r"(?i)\b(?:no steps (?:tried|yet)|I (?:have )?(?:tried|done) nothing|I have not tried anything)\b", sentence):
                result["steps_known"] = True
    if result["ticket_request_id"] == (previous or {}).get("ticket_request_id"):
        result["ticket_id"] = (previous or {}).get("ticket_id")
    return result
