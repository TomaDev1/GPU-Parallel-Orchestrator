import asyncio
import json

import pytest

httpx = pytest.importorskip("httpx")

from orchestrator import OpenAICompatibleBackend, Request  # noqa: E402


def test_sends_chat_completion_and_parses_response():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["auth"] = req.headers.get("authorization")
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        })

    async def main():
        be = OpenAICompatibleBackend(
            "http://srv:8000/v1/", "qwen", api_key="k", transport=httpx.MockTransport(handler)
        )
        try:
            return await be.complete(Request("hi", system="sys", max_tokens=5))
        finally:
            await be.aclose()

    out = asyncio.run(main())
    assert out.text == "hello" and out.prompt_tokens == 7 and out.output_tokens == 3
    assert seen["url"] == "http://srv:8000/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "sys"}, {"role": "user", "content": "hi"},
    ]
    assert seen["body"]["max_tokens"] == 5
