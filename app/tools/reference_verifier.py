import ipaddress
import json
import re
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse, urlunparse

import httpx


MAX_PAGE_BYTES = 1_000_000
MAX_REDIRECTS = 5

USER_AGENT = (
    "Mozilla/5.0 (compatible; QAReferenceBot/1.0; "
    "+https://example.com/qa-reference-verifier)"
)


class PageTextParser(HTMLParser):
    """Extract page title and readable page text."""

    IGNORED_TAGS = {
        "script",
        "style",
        "noscript",
        "svg",
        "nav",
        "footer",
        "header",
    }

    def __init__(self):
        super().__init__()
        self.in_title = False
        self.page_title = ""
        self.text_parts = []
        self.ignored_depth = 0
        self.ignored_tags_stack = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()

        if tag == "title":
            self.in_title = True

        if tag in self.IGNORED_TAGS:
            self.ignored_depth += 1
            self.ignored_tags_stack.append(tag)

    def handle_endtag(self, tag):
        tag = tag.lower()

        if tag == "title":
            self.in_title = False

        if tag in self.IGNORED_TAGS and tag in self.ignored_tags_stack:
            self.ignored_tags_stack.remove(tag)
            self.ignored_depth = max(0, self.ignored_depth - 1)

    def handle_data(self, data):
        value = data.strip()

        if not value or self.ignored_depth > 0:
            return

        if self.in_title:
            self.page_title += " " + value

        self.text_parts.append(value)


STOP_WORDS = {
    "what", "which", "when", "where", "does", "this", "that",
    "with", "from", "into", "about", "their", "there", "have",
    "been", "being", "your", "will", "would", "could", "should",
    "how", "why", "the", "and", "for", "are", "was", "were",
    "its", "has", "can", "not", "use", "used", "using",
    "purpose", "difference", "between", "explain", "describe",
    "define", "modern", "current", "following", "other",
    "forms", "based", "each", "some", "such", "these",
    "those", "also", "than", "then", "they", "them",
    "is", "in", "of", "to", "on", "or", "by", "an", "as",
}

TOPIC_SYNONYMS = {
    "agentic": {
        "agentic",
        "agent",
        "agents",
        "autonomous",
        "autonomy",
        "llm",
        "llms",
        "ai",
    },
    "python": {"python"},
    "machine": {"machine", "learning", "ml"},
    "fastapi": {"fastapi", "api"},
    "django": {"django"},
    "generative": {"generative", "generation", "genai", "llm"},
    "genai": {"genai", "generative", "llm", "llms"},
    "rag": {"rag", "retrieval", "augmented", "generation"},
}


def get_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text."""

    words = re.findall(r"[a-z0-9]+", text.lower())

    return {
        word
        for word in words
        if len(word) > 1 and word not in STOP_WORDS
    }


def normalize_arxiv_url(url: str) -> str:
    """
    Convert arXiv PDF URLs into abstract-page URLs.

    Example:
    https://arxiv.org/pdf/2308.11432.pdf
    becomes:
    https://arxiv.org/abs/2308.11432
    """

    parsed = urlparse(url)

    if parsed.hostname and parsed.hostname.lower() == "arxiv.org":
        path = parsed.path

        match = re.fullmatch(
            r"/pdf/([^/]+?)(?:\.pdf)?",
            path,
            flags=re.IGNORECASE,
        )

        if match:
            paper_id = match.group(1)

            return urlunparse(
                (
                    parsed.scheme,
                    parsed.netloc,
                    f"/abs/{paper_id}",
                    "",
                    "",
                    "",
                )
            )

    return url


def is_public_hostname(hostname: str) -> bool:
    """Reject localhost and private, reserved or unresolvable hosts."""

    hostname = hostname.lower().strip(".")

    if (
        not hostname
        or hostname == "localhost"
        or hostname.endswith(".local")
    ):
        return False

    try:
        addresses = {
            ipaddress.ip_address(hostname)
        }

    except ValueError:
        try:
            resolved = socket.getaddrinfo(
                hostname,
                None,
                type=socket.SOCK_STREAM,
            )

            addresses = {
                ipaddress.ip_address(result[4][0])
                for result in resolved
            }

        except (socket.gaierror, OSError):
            return False

    return bool(addresses) and all(
        address.is_global for address in addresses
    )


def validate_url(url: str) -> tuple[bool, str]:
    """Validate URL scheme, credentials and hostname."""

    try:
        parsed = urlparse(url)

        if (
            parsed.scheme not in ("http", "https")
            or not parsed.netloc
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            return False, "Invalid HTTP or HTTPS URL."

        if not is_public_hostname(parsed.hostname):
            return (
                False,
                "Local, private or unresolvable hostnames are not accepted.",
            )

        return True, ""

    except (ValueError, TypeError):
        return False, "Invalid URL."


def fetch_html_page(url: str) -> dict:
    """
    Fetch an HTML page while validating every redirect destination.

    Redirects are handled manually so that a redirect to a private
    or local hostname is rejected before making the next request.
    """

    current_url = url

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
    }

    with httpx.Client(
        timeout=15.0,
        follow_redirects=False,
        headers=headers,
    ) as client:

        for redirect_count in range(MAX_REDIRECTS + 1):

            is_valid, reason = validate_url(current_url)

            if not is_valid:
                return {
                    "success": False,
                    "reason": reason,
                }

            with client.stream("GET", current_url) as response:

                if response.status_code in {
                    301, 302, 303, 307, 308
                }:
                    location = response.headers.get("location")

                    if not location:
                        return {
                            "success": False,
                            "reason": "Redirect response has no Location header.",
                        }

                    if redirect_count >= MAX_REDIRECTS:
                        return {
                            "success": False,
                            "reason": "Maximum redirect limit exceeded.",
                        }

                    current_url = urljoin(
                        str(response.url),
                        location,
                    )

                    continue

                response.raise_for_status()

                content_type = response.headers.get(
                    "content-type",
                    "",
                ).lower()

                if not (
                    "text/html" in content_type
                    or "application/xhtml+xml" in content_type
                ):
                    return {
                        "success": False,
                        "reason": (
                            f"URL did not return an HTML page. "
                            f"Content-Type: {content_type or 'unknown'}"
                        ),
                    }

                content = bytearray()

                for chunk in response.iter_bytes():
                    remaining = MAX_PAGE_BYTES - len(content)

                    if remaining <= 0:
                        break

                    content.extend(chunk[:remaining])

                    if len(content) >= MAX_PAGE_BYTES:
                        break

                encoding = response.encoding or "utf-8"

                page_html = bytes(content).decode(
                    encoding,
                    errors="replace",
                )

                return {
                    "success": True,
                    "url": str(response.url),
                    "status_code": response.status_code,
                    "content": page_html,
                }

    return {
        "success": False,
        "reason": "Unable to fetch the requested page.",
    }


def verify_reference(
    url: str,
    topic: str = "",
    question: str = "",
) -> dict:
    """
    Check URL accessibility and relevance to the requested topic
    and question using page-content keyword matching.

    This checks accessibility and keyword relevance. It does not
    independently prove that every factual claim is correct.
    """

    if not url or not isinstance(url, str):
        return {
            "verified": False,
            "url": url,
            "reason": "URL is empty or invalid.",
        }

    url = url.strip()

    # Convert arXiv PDF links into HTML abstract-page links.
    url = normalize_arxiv_url(url)

    is_valid, reason = validate_url(url)

    if not is_valid:
        return {
            "verified": False,
            "url": url,
            "reason": reason,
        }

    try:
        result = fetch_html_page(url)

        if not result["success"]:
            return {
                "verified": False,
                "url": url,
                "reason": result["reason"],
            }

        final_url = result["url"]
        status_code = result["status_code"]
        page_html = result["content"]

        parser = PageTextParser()
        parser.feed(page_html)
        parser.close()

        title = parser.page_title.strip()
        page_text = " ".join(parser.text_parts)
        page_content = f"{title} {page_text}"

        if not title or len(page_text) < 100:
            return {
                "verified": False,
                "url": url,
                "status_code": status_code,
                "page_title": title,
                "reason": "Page does not contain enough readable content.",
            }

        topic_words = get_keywords(topic)
        question_words = get_keywords(question)
        page_words = get_keywords(page_content)
        title_words = get_keywords(title)

        # Expand topic keywords using known synonyms.
        topic_anchors = set(topic_words)

        for word in topic_words:
            topic_anchors.update(
                TOPIC_SYNONYMS.get(word, set())
            )

        topic_matches = topic_anchors.intersection(page_words)
        title_topic_matches = topic_anchors.intersection(title_words)

        if not topic_matches:
            return {
                "verified": False,
                "url": url,
                "status_code": status_code,
                "page_title": title,
                "reason": "Page content does not match the requested topic.",
            }

        # Find question-specific keywords after removing topic keywords.
        question_specific_words = question_words - topic_anchors

        question_matches = question_specific_words.intersection(
            page_words
        )

        # General definition questions need a topic match in the title.
        if not question_specific_words:
            if not title_topic_matches:
                return {
                    "verified": False,
                    "url": url,
                    "status_code": status_code,
                    "page_title": title,
                    "topic_matches": sorted(topic_matches),
                    "reason": (
                        "The page title does not support the general "
                        "topic question."
                    ),
                }

            question_matches = title_topic_matches

        elif not question_matches:
            return {
                "verified": False,
                "url": url,
                "status_code": status_code,
                "page_title": title,
                "topic_matches": sorted(topic_matches),
                "reason": "Page does not contain question-specific content.",
            }

        return {
            "verified": True,
            "url": final_url,
            "status_code": status_code,
            "page_title": title,
            "topic_matches": sorted(topic_matches),
            "question_matches": sorted(question_matches),
            "reason": (
                "Accessible page with topic and question keyword matches."
            ),
        }

    except httpx.HTTPError as error:
        return {
            "verified": False,
            "url": url,
            "reason": f"URL verification failed: {error}",
        }

    except Exception as error:
        return {
            "verified": False,
            "url": url,
            "reason": f"Unexpected verification error: {error}",
        }


if __name__ == "__main__":

    print("\n--- RELEVANT REFERENCE TEST ---")

    test_result = verify_reference(
        "https://arxiv.org/abs/2308.11432",
        "Agentic AI",
        "What is Agentic AI?",
    )

    print(json.dumps(test_result, indent=2))