import asyncio
import pytest
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import aiohttp
import random
load_dotenv()


def api_endpoint():
    env = os.environ.get('ENV', 'development')
    if env == 'production':
        return "https://llmapi.ultrasev.com/v2"
    elif env == 'development':
        return "http://192.168.31.46:3000/v2"
    else:
        raise ValueError(f"Invalid environment: {env}")


async def make_request(supplier: str, api_key: str, model: str):
    BASE_URL = api_endpoint() + f"/{supplier}"
    query = "Count from 1 to 5"

    client = AsyncOpenAI(base_url=BASE_URL, api_key=api_key)

    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}],
        stream=True,
    )

    content = ""
    async for chunk in stream:
        content += chunk.choices[0].delta.content or ""

    for i in range(1, 6):
        assert str(i) in content


@pytest.mark.asyncio
async def test_mistral_streaming():
    await make_request(
        supplier="mistral",
        api_key=os.environ["MISTRAL_API_KEY"],
        model="open-mistral-7b"
    )


@pytest.mark.asyncio
async def test_groq_streaming():
    await make_request(
        supplier="groq",
        api_key=os.environ["GROQ_API_KEY"],
        model="llama3-70b-8192"
    )
