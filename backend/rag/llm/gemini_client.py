"""
llm/gemini_client.py
─────────────────────
Wraps the Google Gemini SDK for inference calls.

Key choices:
  • temperature=0.1  — near-deterministic for clinical output
  • max_tokens=8192 — enough for a detailed severity report
  • Retries once on transient API failures before raising
"""
from __future__ import annotations

import time

from google import genai
from google.genai import types

from utils.config import GEMINI_API_KEY, GEMINI_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)


class GeminiClient:
    """Thin wrapper around the Google Gemini API client."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or GEMINI_MODEL
        self._client = None
        if GEMINI_API_KEY:
            self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._temperature = 0.1
        self._max_tokens = 8192
        logger.info(f"GeminiClient initialised with model: {self._model_name}")

    def _ensure_client(self) -> genai.Client:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. Add it to the .env file before running Gemini analysis."
            )
        if self._client is None:
            self._client = genai.Client(api_key=GEMINI_API_KEY)
        return self._client

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        retries: int = 1,
    ) -> str:
        """
        Send *prompt* to Gemini and return the response text.

        Parameters
        ----------
        prompt : str
            The user-turn prompt (contains patient data + retrieved context).
        system_instruction : str, optional
            System-level instruction for the model persona.
        retries : int
            Number of retry attempts on transient failures.

        Returns
        -------
        str
            Raw model response text.

        Raises
        ------
        RuntimeError
            If all retry attempts fail.
        """
        combined_prompt = prompt
        if system_instruction:
            combined_prompt = f"{system_instruction}\n\n{prompt}"

        fallback_models = [self._model_name, "gemini-3.6-flash", "gemini-2.0-flash", "gemini-flash-latest"]
        # deduplicate while preserving order
        unique_models = []
        for m in fallback_models:
            if m and m not in unique_models:
                unique_models.append(m)

        for current_model in unique_models:
            for attempt in range(retries + 1):
                try:
                    client = self._ensure_client()
                    logger.info(
                        f"Gemini request — model={current_model} "
                        f"prompt_len={len(prompt):,} chars (attempt {attempt + 1})"
                    )
                    response = client.models.generate_content(
                        model=current_model,
                        contents=combined_prompt,
                        config=types.GenerateContentConfig(
                            temperature=self._temperature,
                            max_output_tokens=self._max_tokens,
                        ),
                    )
                    text = response.text
                    self._model_name = current_model
                    logger.info(f"Gemini response received ({len(text):,} chars).")
                    return text

                except Exception as exc:
                    exc_str = str(exc)
                    if "404" in exc_str or "NOT_FOUND" in exc_str or "no longer available" in exc_str:
                        logger.warning(f"Model {current_model} not found/deprecated. Trying next fallback model...")
                        break  # Break inner retry loop, try next model in unique_models

                    if attempt < retries:
                        wait = 2 ** attempt  # exponential back-off
                        logger.warning(
                            f"Gemini API error (attempt {attempt + 1}): {exc}. "
                            f"Retrying in {wait}s…"
                        )
                        time.sleep(wait)
                    else:
                        logger.error(f"Gemini API failed after {retries + 1} attempts on {current_model}: {exc}")
                        if current_model == unique_models[-1]:
                            raise RuntimeError(f"Gemini API error: {exc}") from exc

        raise RuntimeError("Failed to generate content with any available Gemini model.")

