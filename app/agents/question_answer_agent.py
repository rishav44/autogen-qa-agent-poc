
import ast
import asyncio
import json
import re

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient

from app.tools.arxiv_search import search_arxiv
from app.tools.reference_verifier import verify_reference


class QuestionAnswerAgent:
    MAX_REFERENCE_ATTEMPTS = 3
    MAX_SEARCH_RESULTS = 10
    MAX_GENERATION_ATTEMPTS = 4

    def __init__(self):
        self.model_client = OllamaChatCompletionClient(
            model="llama3.2:3b",
            host="http://localhost:11434",
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "llama",
                "structured_output": True,
            },
        )

        self.generator_agent = AssistantAgent(
            name="qa_generator",
            model_client=self.model_client,
            system_message=(
                "You are an experienced Python technical educator and "
                "research assistant. Generate relevant, technically "
                "accurate questions and answers. Never invent APIs, "
                "packages, paper details, or code behavior. "
                "Follow the requested JSON schema exactly."
            ),
        )

    @staticmethod
    def extract_json(response):
        """Extract a JSON object from the model response."""

        if isinstance(response, dict):
            return (
                response
                if isinstance(response.get("questions"), list)
                else None
            )

        if not isinstance(response, str):
            return None

        cleaned = response.strip()

        # Remove outer Markdown code fences.
        cleaned = re.sub(
            r"^\s*```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)

        # First try normal JSON parsing.
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and isinstance(
                data.get("questions"), list
            ):
                return data
        except json.JSONDecodeError:
            pass

        # Find a complete JSON object, respecting quoted strings.
        start = cleaned.find("{")

        while start != -1:
            depth = 0
            in_string = False
            escaped = False

            for index in range(start, len(cleaned)):
                char = cleaned[index]

                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue

                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1

                    if depth == 0:
                        candidate = cleaned[start:index + 1]

                        try:
                            data = json.loads(candidate)

                            if (
                                isinstance(data, dict)
                                and isinstance(data.get("questions"), list)
                            ):
                                return data
                        except json.JSONDecodeError:
                            break

            start = cleaned.find("{", start + 1)

        print("JSON parsing failed.")
        print(repr(response))
        return None

    @staticmethod
    def get_response_text(messages):
        for message in reversed(messages):
            content = getattr(message, "content", None)

            if isinstance(content, str) and content.strip():
                return content.strip()

            if isinstance(content, dict):
                return content

        return ""

    @staticmethod
    def normalize_question(question):
        return re.sub(r"\W+", " ", question.lower()).strip()

    @staticmethod
    def extract_python_code(answer):
        """Return Python code blocks from an answer."""

        if not isinstance(answer, str):
            return []

        pattern = r"```(?:python|py)\s*(.*?)```"
        return re.findall(
            pattern,
            answer,
            flags=re.IGNORECASE | re.DOTALL,
        )

    @classmethod
    def validate_answer(cls, answer):
        """Check answer quality and Python code syntax."""

        if not isinstance(answer, str) or len(answer.strip()) < 80:
            return False, "Answer is too short."

        code_blocks = cls.extract_python_code(answer)

        for code in code_blocks:
            if not code.strip():
                return False, "Empty Python code example."

            try:
                ast.parse(code)
            except SyntaxError as error:
                return False, (
                    f"Invalid Python code syntax: {error.msg} "
                    f"at line {error.lineno}"
                )

        return True, "Answer passed basic validation."

    @staticmethod
    def extract_tool_results(messages):
        results = []

        for message in messages:
            if type(message).__name__ != "ToolCallExecutionEvent":
                continue

            content = getattr(message, "content", [])

            if not isinstance(content, list):
                continue

            for execution in content:
                if getattr(execution, "is_error", False):
                    print(
                        "Tool execution error:",
                        getattr(execution, "content", ""),
                    )
                    continue

                raw = getattr(execution, "content", "")

                if isinstance(raw, dict):
                    results.append(raw)
                    continue

                if isinstance(raw, str):
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        try:
                            parsed = ast.literal_eval(raw)
                        except (ValueError, SyntaxError):
                            continue

                    if isinstance(parsed, dict):
                        results.append(parsed)

        return results

    async def search_research_papers(self, topic):
        print("\n========== ARXIV RESEARCH SEARCH ==========")
        print("Searching for:", topic)

        papers = await asyncio.to_thread(
            search_arxiv,
            topic,
            self.MAX_SEARCH_RESULTS,
        )

        if not isinstance(papers, list):
            raise ValueError("arXiv search did not return a list.")

        valid_papers = []

        for paper in papers:
            if not isinstance(paper, dict):
                continue

            title = str(paper.get("title", "")).strip()
            url = str(paper.get("url", "")).strip()
            summary = str(paper.get("summary", "")).strip()

            if not title or not summary or not url:
                continue

            if not url.startswith((
                "https://arxiv.org/abs/",
                "http://arxiv.org/abs/",
            )):
                continue

            valid_papers.append({
                "title": title,
                "url": url,
                "summary": summary,
                "authors": paper.get("authors", []),
                "published": paper.get("published", ""),
                "categories": paper.get("categories", []),
            })

        print("Valid research papers found:", len(valid_papers))

        for index, paper in enumerate(valid_papers, start=1):
            print(f"{index}. {paper['title']}")
            print("   URL:", paper["url"])

        if not valid_papers:
            raise ValueError(
                "No valid arXiv research papers were found for this topic."
            )

        return valid_papers

    async def generate_question_batch(
        self,
        topic,
        complexity,
        required_count,
        existing_questions,
        research_papers,
    ):
        previous = "\n".join(
            f"- {question}" for question in existing_questions
        ) or "None"

        paper_information = [
            {
                "paper_number": index,
                "title": paper["title"],
                "url": paper["url"],
                "summary": paper["summary"][:1500],
            }
            for index, paper in enumerate(research_papers, start=1)
        ]

        papers_json = json.dumps(
            paper_information,
            ensure_ascii=False,
            indent=2,
        )

        complexity_guidance = {
            "Beginner": (
                "Use simple language and foundational concepts. "
                "Avoid unnecessarily specialized research topics. "
                "Explain terminology and use small, understandable examples."
            ),
            "Intermediate": (
                "Cover practical implementation, common design choices, "
                "libraries, trade-offs, and realistic use cases."
            ),
            "Advanced": (
                "Cover architecture, internal mechanisms, performance, "
                "scalability, algorithms, and technical trade-offs."
            ),
        }

        task = f"""
You are generating a technical learning question bank.

TOPIC: {topic}
COMPLEXITY: {complexity}
NUMBER OF QUESTIONS REQUIRED: {required_count}

COMPLEXITY INSTRUCTIONS:
{complexity_guidance[complexity]}

QUESTIONS ALREADY GENERATED:
{previous}

RESEARCH PAPERS:
{papers_json}

QUESTION QUALITY RULES:

1. Every question must be directly relevant to the requested topic.
2. Do not generate unrelated questions just because a paper contains
   the word Python or another matching keyword.
3. Select a paper whose title and summary genuinely relate to the
   question.
4. Do not turn a niche research paper into a general implementation
   question unless the paper supports that topic.
5. Questions must be unique and useful for learning or interviews.
6. Answers must explain the concept, its purpose, and how it works.
7. Include practical examples when they are useful.
8. Python examples must be syntactically valid.
9. Do not invent package names, APIs, functions, or library behavior.
10. Do not claim that an example was tested unless it was actually tested.
11. Do not include code if you are unsure that it is correct.
12. Do not invent research-paper details.

REFERENCE RULES:

1. Select references only from the supplied paper list.
2. Copy the exact URL from the selected paper.
3. Never invent or modify a URL.
4. Use the most relevant paper for each question.

OUTPUT RULES:

Return ONLY valid JSON.
Do not use Markdown fences around the JSON.
Do not add text outside the JSON.

Required structure:

{{
  "questions": [
    {{
      "question": "Question text",
      "answer": "Detailed answer",
      "reference_url": "Exact URL from the supplied papers"
    }}
  ]
}}

The questions array must contain exactly {required_count} objects.
"""

        result = await self.generator_agent.run(task=task)
        raw = self.get_response_text(result.messages)

        print("\n========== MODEL RESPONSE ==========")
        print(raw)
        print("====================================")

        return self.extract_json(raw)

    async def generate_exact_questions(
        self,
        topic,
        complexity,
        question_count,
        research_papers,
    ):
        questions = []
        seen = set()

        allowed_urls = {
            paper["url"] for paper in research_papers
        }

        for attempt in range(1, self.MAX_GENERATION_ATTEMPTS + 1):
            remaining = question_count - len(questions)

            if remaining <= 0:
                break

            print(
                f"\nGeneration attempt {attempt}/"
                f"{self.MAX_GENERATION_ATTEMPTS}"
            )
            print("Questions still required:", remaining)

            data = await self.generate_question_batch(
                topic=topic,
                complexity=complexity,
                required_count=remaining,
                existing_questions=[
                    item["question"] for item in questions
                ],
                research_papers=research_papers,
            )

            if data is None:
                print("Could not parse model response. Retrying...")
                continue

            batch = data.get("questions", [])

            if not isinstance(batch, list):
                continue

            for item in batch:
                if len(questions) >= question_count:
                    break

                if not isinstance(item, dict):
                    continue

                question = str(item.get("question", "")).strip()
                answer = str(item.get("answer", "")).strip()
                url = str(item.get("reference_url", "")).strip()

                if not question or not answer:
                    continue

                if url not in allowed_urls:
                    print("Rejected invented or unlisted URL:", url)
                    continue

                valid, reason = self.validate_answer(answer)

                if not valid:
                    print("Rejected answer:", reason)
                    continue

                normalized = self.normalize_question(question)

                if not normalized or normalized in seen:
                    print("Duplicate question skipped:", question)
                    continue

                seen.add(normalized)

                questions.append({
                    "question": question,
                    "answer": answer,
                    "reference_url": url,
                })

            print(
                f"Questions collected: "
                f"{len(questions)}/{question_count}"
            )

        if len(questions) != question_count:
            raise ValueError(
                f"Could not generate exactly {question_count} valid "
                f"questions. Generated {len(questions)}. "
                "Try another topic or complexity."
            )

        return questions

    async def verify_with_autogen(self, topic, question, url):
        """Use AutoGen to call the Python reference verification tool."""

        def verify_candidate_reference(candidate_url: str) -> dict:
            return verify_reference(
                url=candidate_url,
                topic=topic,
                question=question,
            )

        verifier_agent = AssistantAgent(
            name="reference_verifier_agent",
            model_client=self.model_client,
            tools=[verify_candidate_reference],
            max_tool_iterations=2,
            reflect_on_tool_use=True,
            system_message=(
                "You are a reference verification agent. "
                "You MUST call verify_candidate_reference using the "
                "exact URL provided. Do not modify the URL. "
                "The Python tool result is authoritative."
            ),
        )

        result = await verifier_agent.run(
            task=(
                "Call verify_candidate_reference exactly once "
                "with this URL:\n"
                f"{url}\n"
                "Do not modify the URL."
            )
        )

        request_count = 0
        execution_count = 0

        print("\n--- AUTOGEN TOOL EXECUTION LOG ---")

        for message in result.messages:
            message_type = type(message).__name__
            print("Message type:", message_type)

            if message_type == "ToolCallRequestEvent":
                calls = getattr(message, "content", [])

                if isinstance(calls, list):
                    request_count += len(calls)

                    for call in calls:
                        print(
                            "Tool requested:",
                            getattr(call, "name", "Unknown"),
                        )
                        print(
                            "Arguments:",
                            getattr(call, "arguments", ""),
                        )

            elif message_type == "ToolCallExecutionEvent":
                executions = getattr(message, "content", [])

                if isinstance(executions, list):
                    execution_count += len(executions)

                    for execution in executions:
                        print(
                            "Tool executed:",
                            getattr(execution, "name", "Unknown"),
                        )
                        print(
                            "Tool error:",
                            getattr(execution, "is_error", False),
                        )
                        print(
                            "Tool result:",
                            getattr(execution, "content", ""),
                        )

        print("Tool requests:", request_count)
        print("Tool executions:", execution_count)

        results = self.extract_tool_results(result.messages)

        if not results:
            return {
                "verified": False,
                "url": url,
                "reason": "No Python tool result was received.",
            }

        verification = results[-1]
        verification["verified"] = (
            verification.get("verified") is True
        )

        return verification

    async def verify_with_retries(
        self,
        topic,
        question,
        initial_url,
        research_papers,
    ):
        candidate_urls = [
            paper["url"] for paper in research_papers
        ]

        if initial_url in candidate_urls:
            candidate_urls.remove(initial_url)
            candidate_urls.insert(0, initial_url)

        attempts = 0
        last_reason = "Reference verification failed."

        for candidate_url in candidate_urls:
            if attempts >= self.MAX_REFERENCE_ATTEMPTS:
                break

            attempts += 1

            print(
                f"\nReference verification attempt "
                f"{attempts}/{self.MAX_REFERENCE_ATTEMPTS}"
            )
            print("Candidate URL:", candidate_url)

            try:
                verification = await self.verify_with_autogen(
                    topic=topic,
                    question=question,
                    url=candidate_url,
                )
            except Exception as error:
                verification = {
                    "verified": False,
                    "url": candidate_url,
                    "reason": f"Verification error: {error}",
                }

            if verification.get("verified") is True:
                print("Reference successfully verified!")
                return verification

            last_reason = verification.get(
                "reason",
                "Reference verification failed.",
            )

            print("Reference rejected:", last_reason)

        return {
            "verified": False,
            "reason": (
                f"No reference could be verified after "
                f"{attempts} attempt(s). Last reason: {last_reason}"
            ),
        }

    async def generate_response(
        self,
        topic,
        complexity,
        question_count,
    ):
        try:
            if not isinstance(topic, str) or not topic.strip():
                raise ValueError("Topic cannot be empty.")

            if complexity not in (
                "Beginner",
                "Intermediate",
                "Advanced",
            ):
                raise ValueError(
                    "Complexity must be Beginner, Intermediate, "
                    "or Advanced."
                )

            if (
                not isinstance(question_count, int)
                or isinstance(question_count, bool)
                or question_count < 1
                or question_count > 20
            ):
                raise ValueError(
                    "Question count must be between 1 and 20."
                )

            topic = topic.strip()

            print("\n========== RESEARCH SEARCH ==========")

            research_papers = await self.search_research_papers(topic)

            print("\n========== Q&A GENERATION ==========")

            questions = await self.generate_exact_questions(
                topic=topic,
                complexity=complexity,
                question_count=question_count,
                research_papers=research_papers,
            )

            print(
                f"\nSuccessfully generated "
                f"{len(questions)} questions."
            )

            print("\n========== REFERENCE VERIFICATION ==========")

            output = [
                "AUTOGEN QUESTION ANSWER AGENT",
                "=" * 60,
                f"TOPIC: {topic}",
                f"COMPLEXITY: {complexity}",
                f"TOTAL QUESTIONS: {question_count}",
                f"ARXIV PAPERS FOUND: {len(research_papers)}",
                "",
            ]

            verified_count = 0

            for index, item in enumerate(questions, start=1):
                question = item["question"]
                answer = item["answer"]
                initial_url = item["reference_url"]

                print(
                    f"\nVerifying reference {index}/{question_count}"
                )

                verification = await self.verify_with_retries(
                    topic=topic,
                    question=question,
                    initial_url=initial_url,
                    research_papers=research_papers,
                )

                output.extend([
                    f"QUESTION {index}:",
                    question,
                    "",
                    "DETAILED ANSWER:",
                    answer,
                    "",
                    "REFERENCE VERIFICATION:",
                ])

                if verification.get("verified") is True:
                    verified_count += 1

                    output.extend([
                        "STATUS: VERIFIED",
                        (
                            "VERIFIED REFERENCE: "
                            f"{verification.get('url', initial_url)}"
                        ),
                        (
                            "PAGE TITLE: "
                            f"{verification.get('page_title', 'Not available')}"
                        ),
                    ])
                else:
                    output.extend([
                        "STATUS: NO VERIFIED REFERENCE",
                        "VERIFIED REFERENCE: Not available.",
                        (
                            "REASON: "
                            f"{verification.get('reason', 'Verification failed.')}"
                        ),
                    ])

                output.extend([
                    "",
                    "-" * 60,
                    "",
                ])

            output.append(
                f"VERIFIED REFERENCES: {verified_count}/{question_count}"
            )

            return "\n".join(output)

        except Exception as error:
            return f"Error while generating questions: {error}"


async def main():
    qa_agent = QuestionAnswerAgent()

    try:
        response = await qa_agent.generate_response(
            topic="Agentic AI",
            complexity="Advanced",
            question_count=5,
        )

        print("\n========== FINAL AUTOGEN Q&A OUTPUT ==========\n")
        print(response)

    finally:
        await qa_agent.model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
