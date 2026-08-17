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

        for attempt in range(retries + 1):
            try:
                client = self._ensure_client()
                logger.info(
                    f"Gemini request — model={self._model_name} "
                    f"prompt_len={len(prompt):,} chars (attempt {attempt + 1})"
                )
                response = client.models.generate_content(
                    model=self._model_name,
                    contents=combined_prompt,
                    config=types.GenerateContentConfig(
                        temperature=self._temperature,
                        max_output_tokens=self._max_tokens,
                    ),
                )
                text = response.text
                logger.info(f"Gemini response received ({len(text):,} chars).")
                return text

            except Exception as exc:
                if attempt < retries:
                    wait = 2 ** attempt  # exponential back-off
                    logger.warning(
                        f"Gemini API error (attempt {attempt + 1}): {exc}. "
                        f"Retrying in {wait}s…"
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"Gemini API failed after {retries + 1} attempts: {exc}")
                    raise RuntimeError(f"Gemini API error: {exc}") from exc

        # Unreachable, but satisfies type-checkers
        raise RuntimeError("Unexpected error in GeminiClient.generate()")
