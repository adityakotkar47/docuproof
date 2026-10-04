"""Auto mode: answer with single-shot RAG; if the model finds the retrieved passages insufficient, escalate to the agent.

Questions refused by the score gate (clearly off-topic) are not escalated. Questions that need several documents, or
whose first search misses, are the ones the agent is for: it searches again, once per part of the question.
"""
from __future__ import annotations

from dataclasses import dataclass

from .agent import Agent, AgentResult
from .rag import RAG, Answer, Conversation


@dataclass
class AutoAnswer:
    question: str
    text: str
    refused: bool
    single: Answer
    agent: AgentResult | None = None    # set when the question was escalated

    @property
    def used_agent(self) -> bool:
        return self.agent is not None


class Assistant:
    def __init__(self, rag: RAG, agent: Agent):
        self.rag, self.agent = rag, agent

    def ask(self, question: str, conv: Conversation | None = None) -> AutoAnswer:
        single = self.rag.ask(question, conv)
        if not single.refused or single.gated:
            return AutoAnswer(question, single.text, single.refused, single)
        result = self.agent.run(single.query_used)      # the standalone question when memory rewrote it
        if conv is not None and conv.turns:
            conv.turns[-1] = (question, result.text)    # remember the answer the user actually saw
        return AutoAnswer(question, result.text, result.refused, single, result)
