import logging
from typing import Any, Dict, List

import colorlog
from anymodel import ModelAPI, ModelParams
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

LOG_FORMAT = "%(log_color)s%(levelname)-8s%(reset)s %(message)s"
colorlog.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class ChatBackend:
    async def chat(self, *_, **__) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAIBackend(ChatBackend):
    def __init__(self, model: str):
        self.model = model
        self.client = ModelAPI(
            model_params=ModelParams(
                name=model,
                infer_kwargs={
                    "temperature": 0.9,
                    "max_tokens": 8192,
                },
            ),
        )

    def _clean_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """清理消息格式，移除tool角色消息中的name参数"""
        cleaned_messages = []
        for message in messages:
            if message.get("role") == "tool":
                # 只保留必需的字段：role, tool_call_id, content
                cleaned_message = {
                    "role": message["role"],
                    "tool_call_id": message["tool_call_id"],
                    "content": message["content"],
                }
                cleaned_messages.append(cleaned_message)
            else:
                cleaned_messages.append(message)
        return cleaned_messages

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: str | None = "auto",
        max_tokens: int = 15000,
    ) -> Dict[str, Any]:
        # 清理消息格式
        cleaned_messages = self._clean_messages(messages)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": cleaned_messages,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        resp = await self.client.chat_completion(messages=cleaned_messages, tools=tools)  # type: ignore[arg-type]
        # print(resp)
        raw_calls = getattr(resp, "tool_calls", None)
        tool_calls = None
        tool_calls_id = None
        if raw_calls:

            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in raw_calls
            ]
        print({"content": resp.content, "tool_calls": tool_calls})
        return {"content": resp.content, "tool_calls": tool_calls}
