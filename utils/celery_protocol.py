# Taken from https://github.com/fairinternal/llm_inference/blob/8ae493b909f7a7618d903952c4bf5119df1c735c/shared/protocol.py  # noqa: B950
#
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Classes that define the request and response formats of the model service.
"""

# Allows to type hint a method with the type of the enclosing class
# (see https://stackoverflow.com/a/33533514):
from __future__ import annotations

import json
from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field


class RequestKind(str, Enum):
    instruct = "instruct"
    chat = "chat"
    chat_v2 = "chat_v2"
    raw = "raw"


class ChatMessage(BaseModel):
    # TODO: replace role with enums ?
    role: str
    content: str

    @staticmethod
    def user(c: str):
        return ChatMessage(role="user", content=c)

    @staticmethod
    def assistant(c: str):
        return ChatMessage(role="assistant", content=c)


class Header(BaseModel):
    # Source is what we use to call "role". e.g. user/assistant/system/ipython/etc...
    source: str = Field(key_name="Source")

    # Destination is the target for the message (e.g. one of the valid sources)
    # In general if it does not make sense to put this field we can leave it blank.
    # E.g. for system message no obvious destination so we leave it blank which
    # can be thought of broadcasting the message to "all"
    destination: str = Field(default=None, key_name="Destination")

    # We want the model to be able to signal when it wants to hand control back to the user.
    eot: bool = Field(default=False, key_name="EOT")

    def serialize(self, ignore_default: bool = True) -> str:
        parts = []
        for cls_attr, val in json.loads(self.json()).items():
            schema_props = self.schema()["properties"]
            field_props = schema_props[cls_attr]
            # Ignore header fields that remain at their default value or None
            if (
                ignore_default
                and "default" in field_props
                and field_props["default"] == val
                or val is None
            ):
                continue
            # Lower-case booleans:
            if isinstance(val, bool):
                val = str(val).lower()
            key_name = field_props["key_name"]
            parts.append(f"{key_name}: {val}")
        return "\n".join(parts)

    @staticmethod
    def deserialize(serialized_header: str) -> Header:
        key_to_cls_attr = Header._key_to_cls_attr()
        fields = serialized_header.split("\n")
        params = {}
        for field in fields:
            field_kv = field.split(": ")
            assert (
                len(field_kv) == 2
            ), f"Header field not of the form '<key>: <value>': {field}"
            key, value = field_kv[0].strip(), field_kv[1].strip()
            assert key in key_to_cls_attr, f"Unknown header field key: {key}"
            params[key_to_cls_attr[key]] = value
        return Header.parse_obj(params)

    @staticmethod
    def _key_to_cls_attr():
        return {
            field["key_name"]: class_attr
            for class_attr, field in Header.schema()["properties"].items()
        }


class ChatMessageV2(BaseModel):
    header: Header
    body: Optional[str] = None

    def serialize(self) -> str:
        res = self.header.serialize()
        if self.body is not None:
            res += "\n\n" + self.body
        return res

    @staticmethod
    def deserialize(serialized_msg: str) -> ChatMessageV2:
        parts = serialized_msg.split("\n\n", 1)
        assert len(parts) > 0, f"Missing header in msg: {serialized_msg}"
        header = Header.deserialize(ChatMessageV2._del_beginning_space(parts[0]))
        if len(parts) > 1:
            return ChatMessageV2(
                header=header, body=ChatMessageV2._del_beginning_space(parts[1])
            )
        return ChatMessageV2(header=header)

    @staticmethod
    def _del_beginning_space(s: str):
        return s[1:] if s.startswith(" ") else s


PromptType = Union[str, List[int], List[ChatMessage], List[ChatMessageV2]]


class CompletionRequest(BaseModel):
    kind: RequestKind
    prompt: Union[PromptType, List[PromptType]]
    stream: bool = True
    min_tokens: int = 0
    max_tokens: int = 256
    stop: Optional[Union[str, List[str]]] = None
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    best_of: Optional[int] = None
    logprobs: Optional[float] = None
    echo: bool = False
    top_k: int = 0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    class Config:
        use_enum_values = True  # for request kind serialization


class LogProbs(BaseModel):
    tokens: List[str]
    token_logprobs: List[float]
    text_offset: List[int]
    top_logprobs: Optional[list] = None


class Completion(BaseModel):
    text: str
    index: int
    logprobs: Optional[LogProbs] = None
    finish_reason: str = ""


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    response_id: str
    object: Optional[str] = "text_completion"
    created: int
    model: Optional[str]
    choices: List[Completion]
    usage: Usage


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


class ChatV2CompletionRequest(ChatCompletionRequest):
    messages: List[ChatMessageV2]
