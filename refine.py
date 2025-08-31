import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

assert os.getenv("OPENAI_API_KEY"), "Please set OPENAI_API_KEY (e.g., in your shell or a .env file)."
client = OpenAI()  



def gloss_to_english_llm(
    gloss_tokens: List[str],
    context_hint: Optional[str] = None,
    temperature: float = 0.2,
    model: str = "gpt-4o-mini"
) -> str:
    """Convert a list of gloss tokens into a fluent English sentence using an LLM.

    Parameters
    ----------
    gloss_tokens : list of str
        Tokens from the CV recognizer (e.g., WSASL/ASL/ISL glosses).
    context_hint : str, optional
        Optional short context (topic/domain) to help resolve ambiguous glosses.
    temperature : float
        Lower values → more deterministic output.
    model : str
        OpenAI model to use.

    Returns
    -------
    str
        A single, natural English sentence ending with proper punctuation.
    """
    # 1) Prepare the instruction; we specify constraints to preserve semantics.
    system = (
        "You are a precise sign-language translator. "
        "Convert gloss tokens to a natural English sentence with correct grammar, "
        "articles, tense, and punctuation. Do not add new facts beyond what is implied "
        "by the glosses. Preserve named entities and numbers exactly. Output only the sentence."
    )

    # 2) Provide a couple of tiny few-shot examples to stabilize style.
    examples = [
        {
            "gloss": ["YESTERDAY", "STORE", "I", "GO"],
            "english": "I went to the store yesterday."
        },
        {
            "gloss": ["TODAY", "SCHOOL", "WE", "MEET", "AFTERNOON"],
            "english": "We will meet at school this afternoon."
        }
    ]

    # 3) Compose user content with the current tokens and optional context.
    user_lines = []
    if context_hint:
        user_lines.append(f"Context: {context_hint}")
    user_lines.append("Gloss tokens: " + " ".join(gloss_tokens))
    user_lines.append("Return a single complete English sentence.")
    user_prompt = "\n".join(user_lines)

    # 4) Build an input string combining instruction + few-shots + current request.
    #    Using the Responses API with `input` keeps things simple.
    few_shot_text = "\n\n".join([
        f"Example gloss: {' '.join(ex['gloss'])}\nExample English: {ex['english']}"
        for ex in examples
    ])

    full_input = (
        f"System: {system}\n\n"
        f"{few_shot_text}\n\n"
        f"Now, translate the following into one fluent English sentence.\n"
        f"{user_prompt}"
    )

    # 5) Call the model. Keep temperature low for consistency.
    resp = client.responses.create(
        model=model,
        input=full_input,
        temperature=temperature
    )

    # 6) Extract the plain text. The SDK exposes output_text for convenience.
    sentence = resp.output_text.strip()

    # 7) Ensure final punctuation.
    if not sentence.endswith(('.', '!', '?')):
        sentence += "."
    return sentence


