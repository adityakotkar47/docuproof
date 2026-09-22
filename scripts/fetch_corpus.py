"""Download the arXiv papers used as the retrieval corpus.

The PDFs are not committed (size and licensing); run this once. Each paper is a
well-known source for a topic in generative AI.
"""
import sys
import time
import urllib.request
from pathlib import Path

PAPERS = {
    # Unit 1: foundations
    "1406.2661": "gan",                      # Generative Adversarial Nets
    "1312.6114": "vae",                      # Auto-Encoding Variational Bayes
    "1706.03762": "transformer",             # Attention Is All You Need
    "2006.11239": "ddpm",                    # Denoising Diffusion Probabilistic Models
    "2112.10752": "latent_diffusion",        # High-Resolution Image Synthesis with Latent Diffusion Models
    "1810.04805": "bert",                    # BERT
    "2005.14165": "gpt3",                    # Language Models are Few-Shot Learners
    "2203.02155": "instructgpt",             # Training language models to follow instructions
    "2106.09685": "lora",                    # LoRA
    "2103.00020": "clip",                    # CLIP
    # Unit 2: RAG, agents, tools
    "2005.11401": "rag",                     # Retrieval-Augmented Generation
    "2004.04906": "dpr",                     # Dense Passage Retrieval
    "1908.10084": "sbert",                   # Sentence-BERT
    "2309.07597": "bge",                     # C-Pack (BGE embeddings)
    "2210.03629": "react",                   # ReAct
    "2302.04761": "toolformer",              # Toolformer
    "2310.11511": "self_rag",                # Self-RAG
    "2201.11903": "chain_of_thought",        # Chain-of-Thought Prompting
    "2307.03172": "lost_in_the_middle",      # Lost in the Middle
    # Unit 4: evaluation, hallucination, ethics
    "2202.03629": "hallucination_survey",    # Survey of Hallucination in NLG
    "2309.15217": "ragas",                   # RAGAS
    "2306.05685": "llm_as_judge",            # Judging LLM-as-a-Judge
    "2412.18004": "correctness_not_faithfulness",  # Correctness is not Faithfulness in RAG Attributions
    "2108.07258": "foundation_model_risks",  # On the Opportunities and Risks of Foundation Models
    "2307.09288": "llama2",                  # Llama 2
}


def main(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    for arxiv_id, name in PAPERS.items():
        dest = out / f"{name}.pdf"
        if dest.exists() and dest.stat().st_size > 10_000:
            continue
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "docuproof-course-project/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                dest.write_bytes(r.read())
            print("ok  ", name, dest.stat().st_size // 1024, "KB")
        except Exception as exc:  # noqa: BLE001
            failed.append((arxiv_id, name, str(exc)))
            print("FAIL", name, exc)
        time.sleep(3)  # be polite to arXiv
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "corpus"))
