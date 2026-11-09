# AutoGen Question Answer Agent POC

## 1. Overview

This project generates educational questions and detailed answers based on a user-provided topic, complexity level, and question count.

It uses Microsoft AutoGen with a locally hosted Ollama LLM (`llama3.2:3b`). A Python reference verification tool checks candidate URLs before they are labelled as verified.

## 2. Features

- Accepts topic, complexity (`Beginner`, `Intermediate`, `Advanced`), and question count.
- Generates distinct questions and detailed answers.
- Uses AutoGen `AssistantAgent` with Ollama `llama3.2:3b`.
- Registers the Python `verify_reference` function as an agent tool.
- Checks candidate reference URLs using Python.
- Returns only URLs that pass the verification checks as verified references.
- Exposes the workflow through FastAPI and Swagger UI.
- Supports local LLM execution without requiring a paid OpenAI API key.

## 3. Technology Stack

- Python 3.13
- Microsoft AutoGen 0.7.5
- Ollama
- Llama 3.2 (3B)
- FastAPI
- Uvicorn
- Pydantic
- HTTPX

## 4. Prerequisites

- Python 3.13 or compatible version
- Ollama installed
- `llama3.2:3b` model downloaded

Download the model:

```bash
ollama pull llama3.2:3b
```

## 5. Installation

Run the following commands from the project root directory:

```bash
python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip

python -m pip install -r requirements.txt
```

If Ollama is not already running, start it in a separate terminal:

```bash
ollama serve
```

Verify the downloaded model:

```bash
ollama list
```

## 6. Start the FastAPI Application

From the project root, activate the virtual environment and execute:

```bash
uvicorn app.api.question_answer_api:app --host 127.0.0.1 --port 8000 --reload
```

Open Swagger UI:

http://127.0.0.1:8000/docs

Expand `POST /generate`, click **Try it out**, and provide the following sample request:

```json
{
  "topic": "Agentic AI",
  "complexity": "Advanced",
  "question_count": 3
}
```

Click **Execute** to generate the questions and answers.

A successful API request returns HTTP 200 with a JSON response.

## 7. Run the Agent Directly

The agent can also be executed without starting the FastAPI server:

```bash
python -m app.agents.question_answer_agent
```

## 8. API Endpoints

### GET /

Returns an API health message and the Swagger documentation path.

### POST /generate

Accepts the following request fields:

| Field | Type | Validation |
|---|---|---|
| topic | String | 2–200 characters |
| complexity | String | Beginner, Intermediate, Advanced |
| question_count | Integer | 1–10 |

Returns the generated questions, detailed answers, and reference verification results.

## 9. Reference Verification

The model proposes a candidate reference URL for each generated question.

The Python `verify_reference` function performs checks including:

- URL format and protocol validation.
- Public hostname validation.
- URL accessibility and HTTP response validation.
- HTML content extraction.
- Topic and question keyword relevance checks.

A candidate URL is labelled as verified only when the Python verifier returns `verified: true`.

Failed candidate URLs are marked unavailable and are not presented as verified references.

**Limitation:** The verification process uses accessibility checks and heuristic keyword matching. It does not guarantee that a reference fully supports every factual claim in the generated answer. Generated answers and references should be reviewed before relying on them.

## 10. Project Structure

```text
autogen_qa_agent/
│
├── app/
│   ├── agents/
│   │   └── question_answer_agent.py
│   │
│   ├── api/
│   │   └── question_answer_api.py
│   │
│   └── tools/
│       └── reference_verifier.py
│
├── autogen_qa_poc_deliverables/
│   ├── README.md
│   ├── architecture.md
│   ├── sample_input.json
│   └── sample_output.json
│
├── requirements.txt
└── ...
```

The repository may also contain files from the pre-existing QA automation application. Those files are separate from this Question Answer Agent POC.

## 11. Troubleshooting

**Ollama connection refused**

Ensure Ollama is running and the model host is configured as `http://localhost:11434`.

**Model not found**

Execute:

```bash
ollama pull llama3.2:3b
```

**Import errors**

Activate the project virtual environment and install dependencies:

```bash
source venv/bin/activate
python -m pip install -r requirements.txt
```

**HTTP 422 Validation Error**

Check the JSON field names, complexity capitalization, and question count range.

**Some references are rejected**

A candidate URL may be blocked, inaccessible, or fail the relevance checks. This does not necessarily mean question generation failed. The API can still return HTTP 200 with the verification results.

## 12. Deliverables

- Python source code.
- AutoGen agent implementation.
- Python reference verification tool.
- FastAPI integration.
- README documentation.
- Architecture explanation.
- Sample input JSON.
- Sample output JSON.
