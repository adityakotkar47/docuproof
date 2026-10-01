import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docuproof.config import DEFAULT  # noqa: E402
from docuproof.ingest import Chunk  # noqa: E402
from docuproof.index import HybridIndex, tokenize  # noqa: E402


class FakeEmbedder:
    """Hashed bag-of-words vectors: deterministic, offline, similar texts get similar vectors."""

    settings = DEFAULT

    def passages(self, texts, batch_size=64):
        out = np.zeros((len(texts), 128), dtype="float32")
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % 128] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)

    def query(self, text):
        return self.passages([text])[0]


class FakeReranker:
    def score(self, query, texts):
        q = set(tokenize(query))
        return [float(len(q & set(tokenize(t)))) for t in texts]


class FakeLLM:
    """Returns scripted replies in order; records the messages it was given."""

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def chat(self, messages, **kw):
        self.seen.append(messages)
        return self.replies.pop(0)

    def chat_tools(self, messages, tools, **kw):
        self.seen.append(messages)
        return self.replies.pop(0)


def tool_call(name, args, call_id="c1"):
    import json

    fn = SimpleNamespace(name=name, arguments=json.dumps(args))
    msg = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id=call_id, function=fn)])
    msg.model_dump = lambda exclude_none=True: {"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": fn.arguments}}]}
    return msg


def final_message(text):
    msg = SimpleNamespace(content=text, tool_calls=None)
    msg.model_dump = lambda exclude_none=True: {"role": "assistant", "content": text}
    return msg


CHUNKS = [
    Chunk("a:1:0", "gan", 1, "The generator and discriminator are trained in an adversarial game."),
    Chunk("b:2:0", "vae", 2, "The variational autoencoder maximises the evidence lower bound."),
    Chunk("c:3:0", "rag", 3, "Retrieval-augmented generation retrieves passages and conditions the generator on them."),
    Chunk("c:3:1", "rag", 3, "Error code E-4021 means the retriever returned no passages."),
    Chunk("d:4:0", "ddpm", 4, "Diffusion models learn to reverse a gradual noising process."),
]


@pytest.fixture()
def index():
    return HybridIndex(list(CHUNKS), FakeEmbedder().passages([c.text for c in CHUNKS]), FakeEmbedder())
