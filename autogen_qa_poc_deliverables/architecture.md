# Architecture — AutoGen Question Answer Agent

## 1. High-Level Workflow

```text
Swagger UI / Client
       |
       | POST /generate
       v
FastAPI Endpoint
(app/api/question_answer_api.py)
       |
       | Validates topic, complexity, question_count
       v
QuestionAnswerAgent
(app/agents/question_answer_agent.py)
       |
       | AutoGen AssistantAgent + Local Ollama Model
       | Generates questions, answers and candidate URLs
       v
JSON Response Parsing
       |
       | Validates question count and uniqueness
       v
Python Reference Verifier
(app/tools/reference_verifier.py)
       |
       | URL validation, HTTP request, HTML extraction
       | Topic/question keyword relevance checks
       v
Formatted Result
       |
       | Labels URL as verified only if verified == true
       v
FastAPI JSON Response
```

## 2. Components

### 2.1 FastAPI Layer

File: `app/api/question_answer_api.py`

Responsibilities:

- Provides `GET /` as a basic health endpoint.
- Provides `POST /generate` to generate questions and answers.
- Uses Pydantic to validate topic, complexity and question count.
- Handles exceptions and closes the model client after execution.

### 2.2 AutoGen Agent

File: `app/agents/question_answer_agent.py`

Responsibilities:

- Creates an `OllamaChatCompletionClient` connected to the local `llama3.2:3b` model.
- Creates an AutoGen `AssistantAgent`.
- Registers the Python `verify_reference` function as an agent tool.
- Generates distinct questions, detailed answers and candidate reference URLs.
- Parses the generated JSON response.
- Validates the requested question count and question uniqueness.
- Performs an independent Python reference verification pass before returning results.

### 2.3 Python Reference Verification Tool

File: `app/tools/reference_verifier.py`

Responsibilities:

- Validates URL format and HTTP/HTTPS protocol.
- Rejects local, private and unresolved hostnames.
- Fetches web pages using HTTPX.
- Follows a limited number of redirects.
- Extracts page titles and readable HTML content.
- Checks topic and question keyword relevance.
- Returns a structured verification result containing status, URL, page title and reason.

### 2.4 Final Response

Each generated question includes:

- Question text.
- Detailed answer.
- Reference verification status.
- Verified reference URL, if verification succeeds.
- Rejection reason, if verification fails.

A candidate URL is labelled as verified only when the Python verifier returns `verified: true`.

## 3. AutoGen Tool Integration

The AutoGen `AssistantAgent` is configured with the Python `verify_reference` function in its tool list.

The model is instructed to call the tool for each generated question. Additionally, the application independently calls the Python verifier for each candidate URL before presenting the final result.

This independent verification pass ensures that candidate URLs are checked even if the model does not invoke the registered tool during its AutoGen execution.

For a demonstration, AutoGen run messages or logs can be inspected to identify actual tool-call events.

## 4. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.13 |
| Agent framework | Microsoft AutoGen 0.7.5 |
| LLM | Ollama Llama 3.2 (3B) |
| Model integration | OllamaChatCompletionClient |
| API framework | FastAPI |
| Data validation | Pydantic |
| HTTP requests | HTTPX |
| API documentation | Swagger UI |

## 5. Limitations

- A small local model may generate generic answers or invalid candidate URLs.
- Keyword matching is heuristic and may produce false positives or false negatives.
- URL accessibility does not guarantee source credibility or factual correctness.
- The verifier does not guarantee that a source supports every factual claim in an answer.
- Some websites block automated requests or return bot-protection responses.
