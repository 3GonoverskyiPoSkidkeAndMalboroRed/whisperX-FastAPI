"""WhisperX implementation of speaker assignment service."""

from typing import Any

import pandas as pd
import whisperx

from app.core.logging import logger


class WhisperXSpeakerAssignmentService:
    """
    WhisperX-based implementation of speaker assignment service.

    This service wraps the WhisperX speaker assignment functionality to
    combine diarization results with aligned transcripts.
    """

    def __init__(self) -> None:
        """Initialize the speaker assignment service."""
        self.logger = logger

    def assign_speakers(
        self,
        diarization_segments: pd.DataFrame,
        transcript: dict[str, Any],
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """
        Assign speaker labels to transcript words using WhisperX.

        Args:
            diarization_segments: DataFrame with speaker segments
            transcript: Aligned transcript dictionary
            progress_callback: Callback для обновления прогресса (опционально)

        Returns:
            Dictionary containing transcript with speaker labels
        """
        if progress_callback:
            progress_callback.update_step(30)

        self.logger.info("   👥 Начало назначения спикеров транскрипту")

        result = whisperx.assign_word_speakers(diarization_segments, transcript)

        if progress_callback:
            progress_callback.update_step(100)

        return result  # type: ignore[no-any-return]
