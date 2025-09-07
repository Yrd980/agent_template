import logging

from agentx.agent import Agent
from agentx.config import Config
from agentx.providers.base import Provider, ChatRequest, ChatResponse, StreamDelta


class DummyProvider(Provider):
    name = "dummy"

    def complete(self, req: ChatRequest) -> ChatResponse:  # noqa: ARG002
        return ChatResponse(content="ok", model="dummy")

    def stream(self, req: ChatRequest):  # noqa: ARG002
        yield StreamDelta(content="ok", done=True)


def test_system_message_inserted_first():
    cfg = Config.from_dict({"provider": "openai"})
    agent = Agent(cfg, DummyProvider(config=cfg, logger=logging.getLogger("test")))
    agent.add_system_message("sys")
    agent.add_user_message("hello")
    assert agent.history[0].role == "system"
    assert agent.history[0].content == "sys"

