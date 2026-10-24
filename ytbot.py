"""
YouTube RAG Bot — Streamlit front end.

    streamlit run ytbot.py
"""

from __future__ import annotations

import dataclasses

import streamlit as st

from core.config import Settings
from core.rag import VideoQA, build_openai_qa
from core.transcript import (
    TranscriptError,
    fetch_transcript,
    format_timestamp,
    get_video_id,
    timestamp_url,
    transcript_to_text,
)

st.set_page_config(page_title="YouTube RAG Bot", page_icon="🎬", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def resolve_api_key(default: str) -> str:
    """API key priority: sidebar input > Streamlit secrets > environment / .env."""
    if default:
        return default
    try:
        return st.secrets.get("OPENAI_API_KEY", "")
    except Exception:  # no secrets.toml present
        return ""


@st.cache_data(show_spinner=False, ttl=3600)
def cached_transcript(video_id: str):
    return fetch_transcript(video_id)


def load_video(video_id: str, settings: Settings) -> VideoQA:
    """Build (or reuse) the vector index for a video, cached per session."""
    cache = st.session_state.setdefault("qa_cache", {})
    key = (video_id, settings.chat_model, settings.embedding_model, settings.chunk_size)
    if key not in cache:
        segments = cached_transcript(video_id)
        cache[key] = build_openai_qa(video_id, segments, settings)
    return cache[key]


def reset_for_new_video(video_id: str) -> None:
    if st.session_state.get("video_id") != video_id:
        st.session_state.video_id = video_id
        st.session_state.messages = []
        st.session_state.summary = None
        st.session_state.start_time = 0


def render_sources(sources, video_id: str, msg_index: int) -> None:
    with st.expander(f"Sources ({len(sources)} transcript excerpts)"):
        for j, doc in enumerate(sources):
            start = doc.metadata["start"]
            cols = st.columns([1, 5])
            if cols[0].button(
                f"▶ {format_timestamp(start)}",
                key=f"jump-{msg_index}-{j}",
                help="Jump the player to this moment",
            ):
                st.session_state.start_time = int(start)
                st.rerun()
            cols[1].markdown(
                f"{doc.page_content[:300]}{'…' if len(doc.page_content) > 300 else ''}  \n"
                f"[Open on YouTube at {format_timestamp(start)}]({timestamp_url(video_id, start)})"
            )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
base = Settings()
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "OpenAI API key",
        type="password",
        value=resolve_api_key(base.openai_api_key),
        help="Read from OPENAI_API_KEY in your .env file if set.",
    )
    model_options = sorted({base.chat_model, "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"})
    chat_model = st.selectbox(
        "Chat model", model_options, index=model_options.index(base.chat_model)
    )
    use_memory = st.toggle(
        "Remember conversation",
        value=True,
        help="Rewrites follow-up questions using earlier chat turns before searching.",
    )
    top_k = st.slider("Chunks retrieved per question (k)", 2, 10, min(max(base.top_k, 2), 10))
    size_options = sorted({500, 750, 1000, 1500, 2000, base.chunk_size})
    chunk_size = st.select_slider(
        "Chunk size (characters)", options=size_options, value=base.chunk_size,
        help="Changing this rebuilds the index for the current video.",
    )
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Fetch the transcript\n"
        "2. Split it into time-stamped chunks\n"
        "3. Embed chunks into a FAISS index\n"
        "4. Rewrite follow-ups into standalone questions\n"
        "5. Retrieve the top-k chunks for the question\n"
        "6. The LLM answers, citing timestamps"
    )

settings = dataclasses.replace(
    base, openai_api_key=api_key, chat_model=chat_model, top_k=top_k, chunk_size=chunk_size
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("🎬 YouTube RAG Bot")
st.caption("Summarize any YouTube video and chat with it. Answers cite the moments they came from.")

url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...", key="url")

if not url:
    st.info("Paste a YouTube link to get started.")
    st.stop()

video_id = get_video_id(url)
if not video_id:
    st.error(
        "That doesn't look like a YouTube link. "
        "Try a youtube.com/watch, youtu.be or /shorts/ URL."
    )
    st.stop()
if not settings.openai_api_key:
    st.warning("Add your OpenAI API key in the sidebar (or in a `.env` file) to continue.")
    st.stop()

reset_for_new_video(video_id)

try:
    with st.spinner("Fetching transcript and building the vector index…"):
        qa = load_video(video_id, settings)
except TranscriptError as e:
    st.error(str(e))
    st.stop()
except Exception as e:  # e.g. invalid API key, network errors
    st.error(f"Something went wrong while indexing the video: {e}")
    st.stop()

left, right = st.columns([5, 6], gap="large")

with left:
    st.video(f"https://www.youtube.com/watch?v={video_id}", start_time=st.session_state.start_time)
    duration = qa.segments[-1].end
    m1, m2, m3 = st.columns(3)
    m1.metric("Length", format_timestamp(duration))
    m2.metric("Caption lines", f"{len(qa.segments):,}")
    m3.metric("Indexed chunks", len(qa.chunks))

    with st.container(border=True):
        st.subheader("Summary")
        if st.button("Summarize video", type="primary", use_container_width=True):
            try:
                with st.spinner("Summarizing…"):
                    st.session_state.summary = qa.summarize()
            except Exception as e:
                st.error(f"Summarization failed: {e}")
        if st.session_state.summary:
            st.markdown(st.session_state.summary)

    with st.expander("Full transcript"):
        transcript_text = transcript_to_text(qa.segments)
        st.text_area("Transcript", transcript_text, height=300, label_visibility="collapsed")
        st.download_button(
            "Download transcript (.txt)", transcript_text, file_name=f"{video_id}_transcript.txt"
        )

with right:
    st.subheader("Ask about the video")
    chat_box = st.container(height=560)
    with chat_box:
        if not st.session_state.messages:
            st.caption("Try: *What are the main arguments?* or *When do they talk about pricing?*")
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                if msg.get("search_query"):
                    st.caption(f"🔎 Searched the transcript for: *{msg['search_query']}*")
                st.markdown(msg["content"])
                if msg.get("sources"):
                    render_sources(msg["sources"], video_id, i)

    if question := st.chat_input("Ask a question about this video"):
        # earlier turns, so follow-ups like "tell me more about that" make sense
        history = st.session_state.messages if use_memory else []
        st.session_state.messages.append({"role": "user", "content": question})
        with chat_box:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Searching the transcript…"):
                        stream, result = qa.stream_answer(
                            question, k=settings.top_k, history=history
                        )
                    rewritten = result.search_query.strip() != question.strip()
                    if rewritten:
                        st.caption(f"🔎 Searched the transcript for: *{result.search_query}*")
                    # the answer appears word by word; write_stream returns the full text
                    text = st.write_stream(stream)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": text,
                            "sources": result.sources,
                            "search_query": result.search_query if rewritten else None,
                        }
                    )
                except Exception as e:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"⚠️ Couldn't answer that: {e}"}
                    )
        st.rerun()  # redraw the chat with the sources and ▶ buttons

    if st.session_state.messages and st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
