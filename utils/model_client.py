import os
from anthropic import Anthropic
from openai import OpenAI
from litellm import completion
import time
import asyncio


class BaseLLMClient:
    def chat(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

    async def a_chat(self, *args, **kwargs):
        return await asyncio.to_thread(self.chat, *args, **kwargs)


class OpenaiClient(BaseLLMClient):
    def __init__(self, keys):
        if isinstance(keys, str):
            self.keys = [keys]
        else:
            self.keys = keys
        self.key_id = 0
        self.api_key = self.keys[self.key_id]
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(
        self,
        messages,
        system_message=None,
        return_text=True,
        reduce_length=False,
        return_history=False,
        msg_history=[],
        *args,
        **kwargs,
    ):
        completion = ""
        new_msg = []
        try:
            if len(msg_history) != 0:

                new_msg = (
                    msg_history + [{"role": "assistant", "content": messages}]
                    if isinstance(messages, str)
                    else msg_history + messages
                )

                completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_message},
                        *new_msg,
                    ],
                    *args,
                    **kwargs,
                    timeout=60,
                )
            else:
                if system_message:
                    new_msg = (
                        [{"role": "assistant", "content": messages}]
                        if isinstance(messages, str)
                        else messages
                    )
                    messages = [
                        {"role": "system", "content": system_message},
                        *new_msg,
                    ]
                completion = self.client.chat.completions.create(
                    messages=messages, *args, **kwargs, timeout=60
                )
        except Exception as e:
            if "This model's maximum context length is" in str(e):
                return "ERROR::reduce_length"
            print(e)

        if return_text:
            if completion != "":
                completion = completion.choices[0].message.content

        if return_history:
            new_msg = new_msg + [{"role": "assistant", "content": completion}]
            return completion, new_msg

        return completion, None

    def text(self, *args, return_text=True, reduce_length=False, **kwargs):
        try:
            completion = self.client.completions.create(*args, **kwargs)
        except Exception as e:
            if "This model's maximum context length is" in str(e):
                return "ERROR::reduce_length"
            time.sleep(0.1)
        if return_text:
            completion = completion.choices[0].text
        return completion

    async def a_chat(self, *args, **kwargs):
        return await asyncio.to_thread(self.chat, *args, **kwargs)

    async def a_text(self, *args, **kwargs):
        return await asyncio.to_thread(self.text, *args, **kwargs)


class ClaudeClient(BaseLLMClient):
    def __init__(self, key):

        self.client = Anthropic(api_key=key)

    def chat(
        self,
        messages,
        return_text=True,
        return_history=False,
        msg_history=[],
        max_tokens=300,
        *args,
        **kwargs,
    ):
        system = " ".join(
            [turn["content"] for turn in messages if turn["role"] == "system"]
        )
        messages = [turn for turn in messages if turn["role"] != "system"]
        if len(system) == 0:
            system = None

        if len(msg_history) != 0:
            new_msg_history = msg_history + messages
            completion = self.client.beta.messages.create(
                messages=new_msg_history,
                system=system,
                max_tokens=max_tokens,
                *args,
                **kwargs,
            )

        else:
            completion = self.client.beta.messages.create(
                messages=messages, system=system, max_tokens=max_tokens, *args, **kwargs
            )
        if return_text:
            completion = completion.content[0].text
        if return_history:
            new_msg_history = new_msg_history + [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": completion,
                        }
                    ],
                }
            ]
            return completion, new_msg_history
        return completion, None

    def text(self, max_tokens=None, return_text=True, *args, **kwargs):
        completion = self.anthropic.beta.messages.create(
            max_tokens_to_sample=max_tokens, *args, **kwargs
        )
        if return_text:
            completion = completion.completion
        return completion

    async def a_chat(self, *args, **kwargs):
        return await asyncio.to_thread(self.chat, *args, **kwargs)

    async def a_text(self, *args, **kwargs):
        return await asyncio.to_thread(self.text, *args, **kwargs)


class LitellmClient(BaseLLMClient):
    def __init__(self, key, base_url):
        self.api_key = key
        self.base_url = base_url
        # self.model= f"{provider}/{deployment}"

    def chat(
        self,
        messages,
        model,
        system_message=None,
        msg_history=None,
        return_text=True,
        return_history=False,
        **kwargs,
    ):
        if not model.startswith("azure/"):
            model = f"azure/{model}"

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        full_messages = list(msg_history or []) + messages
        if system_message:
            full_messages = [
                {"role": "system", "content": system_message}
            ] + full_messages

        response = completion(
            model=model,
            messages=full_messages,
            api_base=self.base_url,
            api_key=self.api_key,
            api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
            drop_params=True,
            **kwargs,
        )
        result = response.choices[0].message.content if return_text else response

        history = None
        if return_history:
            history = (list(msg_history or []) + messages) + [
                {"role": "assistant", "content": result}
            ]
        return result, history

    async def a_chat(self, *args, **kwargs):
        return await asyncio.to_thread(self.chat, *args, **kwargs)


def get_llm_client(model_name):
    # if (
    #     model_name.startswith("gpt")
    #     or model_name.startswith("o3-mini")
    #     or model_name.startswith("o1")
    # ):
    #     api_key = os.getenv("OPENAI_API_KEY")
    #     return OpenaiClient(api_key)
    # elif model_name.startswith("claude"):
    #     api_key = os.getenv("ANTHROPIC_API_KEY")
    #     return ClaudeClient(api_key)
    # else:
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    print("Fetched variables from env file\n")
    print(f"API Base configured: {bool(base_url)}")
    print(f"API Key configured: {bool(api_key)}")
    return LitellmClient(api_key, base_url)
    # else:
    #     raise ValueError(f"Unsupported model name: {model_name}")
