# gen-ai

GenAI utilities and helpers used across the `ml-toolkit` monorepo.

Includes:
- OpenAI-compatible client helpers (OpenAI / Ollama / Mistral)
- sync + async completion helpers and streaming utilities
- lightweight settings via Pydantic (`gen_ai.settings`)

## [llm_helpers](./src/gen_ai/llm_helpers.py)

Minimal helper functions for working with LLMs. Supports Ollama, MistralAI, and other OpenAI-compatible providers.

### Client creation

```python
from openai import OpenAI

from gen_ai.llm_helpers import create_mistral_client, create_ollama_client

# Local Ollama
client = create_ollama_client(
	base_url="http://localhost:11434",
)

# OpenAI API
openai_client = OpenAI(
	api_key="openai-api-key",
)

# MistralAI
mistral = create_mistral_client(
	server_url="mistralai-server-url",
	api_key="mistralai-api-key",
)
```

### Basic completion

```python
from gen_ai.llm_helpers import completion

response = completion(
	client,
	model="llama3.2:1b",
	prompt="What is 2+2?",
	temperature=0.7,
)
print(response)
```

### Structured output

```python
from pydantic import BaseModel, Field

from gen_ai.llm_helpers import completion


class Answer(BaseModel):
	result: int
	explanation: str
	confidence: float = Field(ge=0, le=1)


answer = completion(
	client,
	model="llama3.2:1b",
	prompt="What is 15 * 23? Show your work.",
	response_model=Answer,
)

print(answer.result)
print(answer.explanation)
print(answer.confidence)
```

### Chat completion

```python
from gen_ai.llm_helpers import chat_completion

messages = [
	{"role": "system", "content": "You are a helpful math tutor."},
	{"role": "user", "content": "What is 2+2?"},
]

response = chat_completion(
	client,
	model="llama3.2:1b",
	messages=messages,
)
print(response)
```

### Streaming

```python
from gen_ai.llm_helpers import stream_completion

messages = [{"role": "user", "content": "Write a haiku about Python."}]

for chunk in stream_completion(client, model="llama3.2:1b", messages=messages):
	if isinstance(chunk, str):
		print(chunk, end="", flush=True)
	else:
		# Final ChatCompletion object
		print("\n\nFull response:")
		print(chunk.choices[0].message.content)
```

### Model listing

```python
from gen_ai.llm_helpers import create_ollama_client, list_models

# list_models is async-only for speed reasons
async_client = create_ollama_client(
    base_url="http://localhost:11434",
    async_client=True,
)

# List available models
models = await list_models(async_client)
print(models)

# List with capabilities (Ollama only)
models_info = await list_models(
    async_client,
    base_url="http://localhost:11434",
    include_capabilities=True,
)
print(models_info)
```
