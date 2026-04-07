from src import config
from src.llm import get_llm
from src.retriever import Retriever


class RAGPipeline:
    def __init__(
        self,
        llm_backend: str | None = None,
        llm_model: str | None = None,
    ) -> None:
        self.retriever = Retriever()
        backend = llm_backend if llm_backend is not None else config.LLM_BACKEND
        self.llm = get_llm(backend, model_name=llm_model)

    def answer(self, query: str, k: int = 5) -> dict:
        retrieved = self.retriever.retrieve(query, k=k)

        context = "\n\n".join(
            [
                f"[Chunk {i+1}] {row['text']}"
                for i, (_, row) in enumerate(retrieved.iterrows())
            ]
        )

        prompt = f"""
You are answering questions using only the provided review excerpts.

Instructions:
- Use only the provided context.
- If the context is insufficient, say so.
- Summarize the evidence clearly.
- Do not invent facts.

Context:
{context}

Question:
{query}

Answer:
""".strip()

        answer = self.llm.generate(prompt)

        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": retrieved,
        }