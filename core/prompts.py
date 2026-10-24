"""Prompt templates for summarization and question answering."""

from langchain_core.prompts import ChatPromptTemplate

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You summarize YouTube video transcripts. Write a concise, informative summary "
            "that captures the main points. Start with a one-paragraph overview, then list "
            "3-6 key takeaways as bullet points. Ignore timestamps and filler words.",
        ),
        ("human", "Summarize this video transcript:\n\n{transcript}"),
    ]
)

COMBINE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are given summaries of consecutive sections of one YouTube video. Merge them "
            "into a single summary: a one-paragraph overview followed by 3-6 key takeaways as "
            "bullet points. Remove repetition.",
        ),
        ("human", "{transcript}"),
    ]
)

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite follow-up questions about a YouTube video into standalone search "
            "queries. Use the chat history to resolve references like 'it', 'that', 'the "
            "second one' or 'tell me more'. Keep the user's meaning and wording where you can. "
            "If the question already stands on its own, return it unchanged. Reply with the "
            "rewritten question only, no explanation.",
        ),
        ("human", "Chat history:\n{history}\n\nFollow-up question: {question}"),
    ]
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about a YouTube video using only the transcript excerpts "
            "provided. Each excerpt is labelled with the time range where it appears, like "
            "[2:15-2:48]. Cite the relevant start time in brackets, e.g. [2:15], next to the "
            "claims it supports. If the excerpts don't contain the answer, say you couldn't "
            "find it in the video rather than guessing.",
        ),
        ("human", "Transcript excerpts:\n{context}\n\n{history}Question: {question}"),
    ]
)
