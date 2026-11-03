# Test-only images

Images under `infrastructure/test/` exist for tests and E2E stacks only. They are never referenced by
`docker-compose.yml` or `docker-compose.prod.yml`.

## fake-llm

Standard-library LLM test double. Build context: `infrastructure/test/fake-llm`. Port from `FAKE_LLM_PORT` (default 8080).

- `POST /v1/chat/completions`: OpenAI `chat.completion` shape.
- `POST /model/{model_id}/converse`: bedrock-runtime Converse shape (SigV4 headers are ignored and never recorded).
- `POST /__control/scenario {mode, match?, content?}`: modes `ok`, `echo`, `error`, `timeout`, `crisis_intent`, `reply`.
  `match` scopes a scenario to requests whose last user message contains it (case-insensitive).
- `POST /__control/reset`, `GET /__control/requests`, `GET /health`.
- `FAKE_LLM_TIMEOUT_SECONDS` (default 65) controls the `timeout` mode sleep.

Dev-profile stacks reach the Converse route with `LLM_PROVIDER=bedrock` and `BEDROCK_ENDPOINT_URL=http://fake-llm:8080`.
