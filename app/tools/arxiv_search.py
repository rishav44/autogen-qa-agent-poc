import re
import time
import xml.etree.ElementTree as ET

import httpx


ARXIV_API_URL = "https://export.arxiv.org/api/query"

ATOM_NAMESPACE = {
    "atom": "http://www.w3.org/2005/Atom",
}

USER_AGENT = "QAReferenceAgent/1.0"

MAX_RETRIES = 3

REQUEST_TIMEOUT = httpx.Timeout(
    connect=15.0,
    read=60.0,
    write=15.0,
    pool=15.0,
)

RETRY_STATUS = {429, 500, 502, 503, 504}


def clean_text(value: str) -> str:
    """Normalize whitespace in text."""
    return " ".join((value or "").split())


def build_search_query(topic: str) -> str:
    """Build an arXiv search query from the requested topic."""

    topic = clean_text(topic)

    if not topic:
        raise ValueError("Topic cannot be empty.")

    topic = re.sub(r"[^\w\s-]", " ", topic)
    topic = clean_text(topic)

    return f'all:"{topic}"'


def search_arxiv(
    query: str,
    max_results: int = 5,
) -> list[dict]:
    """
    Search arXiv and return real research-paper references.

    Each result contains:
        title
        url
        summary
        authors
        published
        categories
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if (
        not isinstance(max_results, int)
        or isinstance(max_results, bool)
        or not 1 <= max_results <= 20
    ):
        raise ValueError("max_results must be between 1 and 20.")

    params = {
        "search_query": build_search_query(query),
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/atom+xml",
    }

    # Force IPv4 for the outgoing connection.
    transport = httpx.HTTPTransport(
        local_address="0.0.0.0",
        retries=2,
    )

    try:
        response = None

        with httpx.Client(
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
            headers=headers,
            transport=transport,
        ) as client:

            for attempt in range(MAX_RETRIES):

                try:
                    response = client.get(
                        ARXIV_API_URL,
                        params=params,
                    )

                except (
                    httpx.TimeoutException,
                    httpx.ConnectError,
                ):
                    if attempt == MAX_RETRIES - 1:
                        raise

                    wait_time = 5 * (attempt + 1)

                    print(
                        f"Connection issue. Retrying in "
                        f"{wait_time} seconds..."
                    )

                    time.sleep(wait_time)
                    continue

                if response.status_code in RETRY_STATUS:

                    if attempt == MAX_RETRIES - 1:
                        response.raise_for_status()

                    retry_after = response.headers.get("Retry-After")

                    try:
                        wait_time = (
                            int(retry_after)
                            if retry_after
                            else 5 * (attempt + 1)
                        )
                    except ValueError:
                        wait_time = 5 * (attempt + 1)

                    # Respect arXiv's request interval.
                    wait_time = min(max(wait_time, 3), 60)

                    print(
                        f"arXiv returned HTTP {response.status_code}. "
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                break

        if response is None:
            raise RuntimeError(
                "No response received from arXiv API."
            )

        root = ET.fromstring(response.content)

        results = []

        for entry in root.findall(
            "atom:entry",
            ATOM_NAMESPACE,
        ):

            title = clean_text(
                entry.findtext(
                    "atom:title",
                    "",
                    ATOM_NAMESPACE,
                )
            )

            summary = clean_text(
                entry.findtext(
                    "atom:summary",
                    "",
                    ATOM_NAMESPACE,
                )
            )

            url = clean_text(
                entry.findtext(
                    "atom:id",
                    "",
                    ATOM_NAMESPACE,
                )
            )

            published = clean_text(
                entry.findtext(
                    "atom:published",
                    "",
                    ATOM_NAMESPACE,
                )
            )

            authors = [
                clean_text(
                    author.findtext(
                        "atom:name",
                        "",
                        ATOM_NAMESPACE,
                    )
                )
                for author in entry.findall(
                    "atom:author",
                    ATOM_NAMESPACE,
                )
            ]

            categories = [
                category.attrib.get("term", "")
                for category in entry.findall(
                    "atom:category",
                    ATOM_NAMESPACE,
                )
            ]

            if not title or not summary or not url:
                continue

            results.append({
                "title": title,
                "url": url,
                "summary": summary,
                "authors": authors,
                "published": published,
                "categories": categories,
            })

        return results

    except httpx.HTTPStatusError as error:
        raise RuntimeError(
            f"arXiv API returned HTTP "
            f"{error.response.status_code}."
        ) from error

    except httpx.TimeoutException as error:
        raise RuntimeError(
            "arXiv API request timed out after retries."
        ) from error

    except httpx.RequestError as error:
        raise RuntimeError(
            f"Unable to connect to arXiv API: {error}"
        ) from error

    except ET.ParseError as error:
        raise RuntimeError(
            f"Unable to parse arXiv API response: {error}"
        ) from error


if __name__ == "__main__":

    topic = "Agentic AI"

    print(f"\nSearching arXiv for: {topic}\n")

    try:
        papers = search_arxiv(
            query=topic,
            max_results=5,
        )

        if not papers:
            print("No research papers found.")

        else:
            print(f"Found {len(papers)} research papers.")

            for index, paper in enumerate(papers, start=1):

                print(f"\n{'=' * 60}")
                print(f"Paper {index}")
                print(f"{'=' * 60}")

                print(f"Title: {paper['title']}")
                print(f"URL: {paper['url']}")
                print(f"Published: {paper['published']}")

                print(
                    f"Authors: {', '.join(paper['authors'])}"
                )

                print(
                    f"Categories: {', '.join(paper['categories'])}"
                )

                print(
                    f"Summary: {paper['summary'][:300]}..."
                )

    except (RuntimeError, ValueError) as error:
        print(f"\nERROR: {error}")
