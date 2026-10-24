"""The RAG pipeline: embed transcript chunks into FAISS, retrieve, and generate with an LLM."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import Settings
from core.prompts import COMBINE_PROMPT, CONDENSE_PROMPT, QA_PROMPT, SUMMARY_PROMPT
from core.transcript import Segment, build_chunks, format_timestamp, transcript_to_text

# One chat turn, as stored by the UI: {"role": "user" | "assistant", "content": "..."}
Message = dict


@dataclass
class Answer:
    text: str
    sources: list[Document] = field(default_factory=list)
    # The question actually used for retrieval (a rewrite of follow-ups, else the original)
    search_query: str = ""


def format_history(history: list[Message], max_messages: int, max_chars: int = 600) -> str:
    """Render the last few chat turns as plain text, trimming long answers."""
    lines = []
    for msg in history[-max_messages:] if max_messages > 0 else []:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = " ".join(str(msg.get("content", "")).split())
        if len(content) > max_chars:
            content = content[:max_chars] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def format_context(docs: list[Document]) -> str:
    """Label each retrieved chunk with its time range so the LLM can cite it."""
    ordered = sorted(docs, key=lambda d: d.metadata.get("start", 0))
    return "\n\n".join(
        f"[{format_timestamp(d.metadata['start'])}-{format_timestamp(d.metadata['end'])}] "
        f"{d.page_content}"
        for d in ordered
    )


class VideoQA:
    """Holds one video's transcript and vector index, and answers questions about it.

    The LLM and embedding model are injected, so tests can pass in fakes and you can
    swap OpenAI for any other LangChain-compatible provider.
    """

    def __init__(
        self,
        video_id: str,
        segments: list[Segment],
        llm: BaseChatModel,
        embeddings: Embeddings,
        settings: Settings | None = None,
    ):
        if not segments:
            raise ValueError("Cannot build a VideoQA from an empty transcript.")
        self.video_id = video_id
        self.segments = segments
        self.llm = llm
        self.settings = settings or Settings()
        self.chunks = build_chunks(
            segments,
            video_id,
            chunk_size=self.settings.chunk_size,
            overlap_segments=self.settings.chunk_overlap_segments,
        )
        self.vector_store = FAISS.from_documents(self.chunks, embeddings)
        self._parser = StrOutputParser()

    # -- retrieval -----------------------------------------------------------
    def retrieve(self, question: str, k: int | None = None) -> list[Document]:
        k = k or self.settings.top_k
        return self.vector_store.similarity_search(question, k=min(k, len(self.chunks)))

    # -- generation ----------------------------------------------------------
    def condense_question(self, question: str, history: list[Message]) -> str:
        """Rewrite a follow-up into a standalone question so retrieval finds the right chunks.

        "What about the second one?" on its own matches nothing useful in the index;
        rewritten with the chat history it becomes a specific, searchable question.
        """
        history_text = format_history(history, self.settings.max_history_messages)
        if not history_text:
            return question
        chain = CONDENSE_PROMPT | self.llm | self._parser
        rewritten = chain.invoke({"history": history_text, "question": question}).strip()
        return rewritten or question

    def answer(
        self, question: str, k: int | None = None, history: list[Message] | None = None
    ) -> Answer:
        history = history or []
        search_query = self.condense_question(question, history) if history else question
        docs = self.retrieve(search_query, k)

        history_text = format_history(history, self.settings.max_history_messages)
        chain = QA_PROMPT | self.llm | self._parser
        text = chain.invoke(
            {
                "context": format_context(docs),
                "history": f"Recent conversation:\n{history_text}\n\n" if history_text else "",
                "question": question,
            }
        )
        return Answer(
            text=text,
            sources=sorted(docs, key=lambda d: d.metadata["start"]),
            search_query=search_query,
        )

    def summarize(self) -> str:
        text = transcript_to_text(self.segments, with_timestamps=False)
        summary_chain = SUMMARY_PROMPT | self.llm | self._parser
        limit = self.settings.max_single_pass_chars
        if len(text) <= limit:
            return summary_chain.invoke({"transcript": text})

        # Map-reduce for very long videos: summarize sections, then merge
        section_size = max(limit // 2, 1)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=section_size, chunk_overlap=min(500, section_size // 10)
        )
        sections = splitter.split_text(text)
        partials = summary_chain.batch([{"transcript": s} for s in sections])
        combine_chain = COMBINE_PROMPT | self.llm | self._parser
        return combine_chain.invoke({"transcript": "\n\n---\n\n".join(partials)})


def build_openai_qa(video_id: str, segments: list[Segment], settings: Settings) -> VideoQA:
    """Convenience constructor wiring VideoQA to OpenAI models."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    llm = ChatOpenAI(
        model=settings.chat_model,
        temperature=0,
        max_tokens=900,
        api_key=settings.openai_api_key,
    )
    embeddings = OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
    return VideoQA(video_id, segments, llm, embeddings, settings)
