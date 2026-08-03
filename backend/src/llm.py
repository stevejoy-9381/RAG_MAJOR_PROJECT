"""
src/llm.py — LLM Configuration and Prompt Template
────────────────────────────────────────────────────
WHAT THIS FILE DOES:
  Sets up the Groq LLM client and defines the prompt template that tells
  the LLM exactly how to behave when answering questions from document chunks.

WHY A SEPARATE FILE?
  Prompt engineering is its own concern. Keeping it separate means:
    - You can swap LLM providers without touching retriever.py
    - You can iterate on the prompt without touching anything else
    - Interviewers see clean architecture when they read your code

CONNECTIONS:
  → Imported by src/retriever.py which passes the LLM and prompt
    into the LangChain RetrievalQA chain.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Load GROQ_API_KEY from .env file
load_dotenv()


from src.config import (
    GROQ_API_KEY, GROQ_MODEL, MODEL_NAME,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P,
)


def get_llm() -> ChatGroq:
    """
    Create and return a Groq LLM client using centralized configuration.

    Model: qwen/qwen3.6-27b on Groq Cloud LPU.
    """
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. "
            "Did you set it in your .env file?"
        )

    model_name = GROQ_MODEL
    temperature = LLM_TEMPERATURE

    try:
        llm = ChatGroq(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=LLM_MAX_TOKENS,
        )
        print("Groq Provider Connected")
        print(f"Model: {model_name}")
        return llm

    except Exception as e:
        print(f"[LLM ERROR] Failed to initialize ChatGroq ({model_name}): {e}")
        raise RuntimeError(f"Failed to initialize Groq LLM provider: {e}") from e



def get_prompt_template() -> PromptTemplate:
    """
    Build and return the prompt template for grounded Q&A.

    WHAT IS A PROMPT TEMPLATE?
      A PromptTemplate is a reusable string with {placeholders}.
      LangChain fills in {context} (retrieved chunks) and {question}
      (user's question) before sending to the LLM.

    WHY THIS SPECIFIC PROMPT DESIGN?

      1. "Use ONLY the context below"
         → Prevents hallucination. The LLM is instructed not to use its
           training data — only what we retrieved from the document.

      2. 'If the answer is not in the context, say "I don't know"'
         → This is critical. Without this line, the LLM invents an answer.
           With it, the system admits uncertainty. Trust > fake confidence.

      3. "Be concise and cite the relevant section if possible."
         → Encourages source grounding in the answer text itself.

      4. "Context:" before the chunks
         → Clear signal to the model about where the document text starts.

    THE {context} PLACEHOLDER:
      LangChain automatically fills this with the 4 retrieved chunks,
      joined together as one string. Each chunk includes its metadata
      (page number, source) so the model can reference them.

    THE {question} PLACEHOLDER:
      Filled with whatever the user typed in the chat input.
    """
    template = """You are a precise document Q&A assistant.

Your job is to answer the user's question using ONLY the context provided below.
Do NOT use any knowledge from outside this context.
If the answer is not present in the context, respond with:
"I don't know — the answer is not in the provided document."

Context:
────────────────────────────────────
{context}
────────────────────────────────────

Question: {question}

Instructions:
- Answer directly and concisely.
- Answer in the same language the question was asked in, when possible, even if the source context is in a different language.
- If the answer spans multiple sections, combine them clearly.
- If you reference a specific fact, mention which part of the context it came from.
- Do not make up information.

Answer:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )
    return prompt
