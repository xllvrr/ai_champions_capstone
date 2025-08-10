import os
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_completion(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: int = 0,
    max_completion_tokens: int = 1024,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )
    return response.choices[0].message.content


def get_completion_by_messages(
    messages: list[ChatCompletionMessageParam],
    model: str = "gpt-4o-mini",
    temperature: int = 0,
    max_completion_tokens: int = 1024,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )
    return response.choices[0].message.content
