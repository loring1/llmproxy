import asyncio
import pytest
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import aiohttp
load_dotenv()


def api_endpoint():
    env = os.environ.get('ENV', 'development')
    if env == 'production':
        return "https://llmapi.ultrasev.com/v2"
    return "http://192.168.31.46:3000/v2"


async def make_request(api_key: str,
                       model: str,
                       supplier: str,
                       query: str = "what is the result of 2*21"):
    BASE_URL = api_endpoint() + "/" + supplier
    client = AsyncOpenAI(base_url=BASE_URL, api_key=api_key)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant。"},
            {"role": "user", "content": query}
        ],
        temperature=0.7,
        top_p=1,
        max_tokens=20
    )
    print(type(response), response)
    return response.choices[0].message.content


@pytest.mark.asyncio
async def test_groq():
    response = await make_request(
        api_key=os.environ["GROQ_API_KEY"],
        model="llama3-70b-8192",
        supplier="groq"
    )
    assert '42' in response


@pytest.mark.asyncio
async def test_gemini():
    response = await make_request(
        api_key=os.environ["GEMINI_API_KEY"],
        model="gemini-1.5-flash",
        supplier="gemini"
    )
    assert '42' in response


@pytest.mark.asyncio
async def test_gpt():
    response = await make_request(
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-3.5-turbo-1106",
        supplier="openai"
    )
    assert '42' in response


# @pytest.mark.asyncio
# async def test_cache_for_openai():
#     results = []
#     for _ in range(5):
#         response = await make_request(
#             api_key=os.environ["OPENAI_API_KEY"],
#             model="gpt-3.5-turbo-1106",
#             supplier="openai",
#             query="7 个小矮人的名字是什么？"
#         )
#         results.append(response)
#     assert len(set(results)) < len(results)


@pytest.mark.asyncio
async def test_mistral():
    response = await make_request(
        api_key=os.environ["MISTRAL_API_KEY"],
        model="open-mistral-7b",
        supplier="mistral"
    )
    assert '42' in response


@pytest.mark.asyncio
async def test_mistral_streaming():
    BASE_URL = api_endpoint() + "/mistral"
    api_key = os.environ["MISTRAL_API_KEY"]
    model = "open-mistral-7b"
    query = "Count from 1 to 5"
    await asyncio.sleep(1)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            BASE_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": query}],
                "stream": True,
            },
            headers={"Authorization": f"Bearer {api_key}"}
        ) as response:
            assert response.status == 200
            content = ""
            async for line in response.content:
                if line.startswith(b"data: "):
                    content += line.decode("utf-8").replace("data: ", "")

            assert "1" in content
            assert "2" in content
            assert "3" in content
            assert "4" in content
            assert "5" in content
