一个基于 [cloudflare worker](https://workers.cloudflare.com/) 的 LLM API 反向代理，支持 OpenAI, Gemini, Groq 等平台。接口兼容 OpenAI API 规范，可以直接使用 OpenAI SDK 调用。

<img src="https://s3.bmp.ovh/imgs/2024/04/29/055ddd90de65037e.png" width="789">

# Quick start

Demo API: `https://llmapi.ultrasev.com`

## 通过路由参数指定平台

通过 `/v2/openai` 路由参数区分不同平台，可结合 `chatbox`、[`沉浸式翻译`](https://immersivetranslate.com/en/) 等插件使用。支持的平台有：

- OpenAI, `https://llmapi.ultrasev.com/v2/openai`
- Gemini, `https://llmapi.ultrasev.com/v2/gemini`
- Groq, `https://llmapi.ultrasev.com/v2/groq`
- Mistral, `https://llmapi.ultrasev.com/v2/mistral`

### Gemini API 示例

```python
from openai import OpenAI
api_key = "YOUR_GEMINI_API_KEY"
model = "gemini-1.5-flash"

BASE_URL = "https://llmapi.ultrasev.com/v2/gemini"

client = OpenAI(base_url=BASE_URL, api_key=api_key)
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant。"},
        {"role": "user", "content": "what is the meaning of life?"}
    ],
    temperature=0.7,
    top_p=1,
    max_tokens=1024,
    top_logprobs=32 # topK
)
print(response.choices[0].message.content)
```

### Mistral API 示例

```python
from openai import OpenAI
api_key = "YOUR_MISTRAL_API_KEY"
model = "open-mistral-7b"

BASE_URL = "https://llmapi.ultrasev.com/v2/mistral"

client = OpenAI(base_url=BASE_URL, api_key=api_key)
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant。"},
        {"role": "user", "content": "what is the meaning of life?"}
    ],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content)

```

# 部署自己的 Worker

创建一个新的 cloudflare worker，把 [api/llm_api_proxy.js](./api/llm_api_proxy.js) 中的代码粘贴进去，然后部署就可以。

配置自定义域名：设置 -> Triggers -> Custom Domains，把你的域名填进去，然后点击 add custom domain 即可。

<img src="https://s3.bmp.ovh/imgs/2024/04/29/e64e6a2787183c26.png" width="789">

## YouTuBe 教程

- 使用 Cloudflare Worker 搭建 LLM API 反向代理

<a href="https://www.youtube.com/watch?v=rfn3lBC11Dk">
    <img src="https://img.youtube.com/vi/rfn3lBC11Dk/0.jpg" alt="使用 Cloudflare Worker 搭建反向代理" width="320" height=200 style="border-radius: 15px;">
</a>

## Notes

- `workers.dev` 域名在国内无法直接访问，如果要在国内直接使用，需要自己配置域名。
- 代码中对 Groq 的代理通过 [vercel.com](https://www.vercel.com) + FastAPI 又套了一层，仅做请求中转，不会保存任何数据，介意的话可以自己部署。
- 暂不支持流式输出，有大佬空闲了修改完可以 pull request。

# Q & A

## 1. Gemini API 无法使用

经常有小伙伴反馈 Groq 和 OpenAI 没有问题，但是 Gemini API 无法使用，一个可能的原因是：cloudflare 中转请求时会根据请求者 IP 按就近原则转发请求，而有的地区不被 Google AI 支持，比如香港，这样就会导致请求返回 `User location is not supported for the API use` 的结果。解决方法是直接使用国内 IP 或者使用其他地区代理，比如美区。

参考 issues：

- https://github.com/ultrasev/llmproxy/issues/11
- https://github.com/ultrasev/llmproxy/issues/4

# 更新日志

请访问 [CHANGELOG](CHANGELOG.md)
