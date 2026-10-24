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
from core.prompts import COMBINE_PROMPT, QA_PROMPT, SUMMARY_PROMPT
from core.transcript import Segment, build_chunks, format_timestamp, transcript_to_text


@dataclass
class Answer:
    text: str
    sources: list[Document] = field(default_factory=list)


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
    def answer(self, question: str, k: int | None = None) -> Answer:
        docs = self.retrieve(question, k)
        chain = QA_PROMPT | self.llm | self._parser
        text = chain.invoke({"context": format_context(docs), "question": question})
        return Answer(text=text, sources=sorted(docs, key=lambda d: d.metadata["start"]))

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