"""WhisperX implementation of alignment service."""

import gc
from typing import Any

import numpy as np
import torch
from whisperx import align, load_align_model

from app.core.exceptions import AudioProcessingError
from app.core.logging import logger
from app.utils.progress import ModelLoadingProgress


class WhisperXAlignmentService:
    """
    WhisperX-based implementation of alignment service.

    This service wraps the WhisperX alignment functionality to align
    transcripts to audio with precise word-level timestamps.
    """

    def __init__(self) -> None:
        """Initialize the alignment service."""
        self.model: Any = None
        self.metadata: Any = None
        self.logger = logger

    def align(
        self,
        transcript: list[dict[str, Any]],
        audio: np.ndarray[Any, np.dtype[np.float32]],
        language_code: str,
        device: str,
        align_model: str | None = None,
        interpolate_method: str = "nearest",
        return_char_alignments: bool = False,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """
        Align transcript to audio using WhisperX alignment.

        Args:
            transcript: List of transcript segments to align
            audio: Audio data as numpy array (float32)
            language_code: Language code of the transcript
            device: Device to use ('cpu' or 'cuda')
            align_model: Specific alignment model to use (optional)
            interpolate_method: Method for handling non-aligned words
            return_char_alignments: Whether to return character-level alignments
            progress_callback: Callback для обновления прогресса (опционально)

        Returns:
            Dictionary containing aligned transcript
        """
        # Log GPU memory before loading model
        if torch.cuda.is_available():
            self.logger.debug(
                f"GPU memory before loading model - used: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, "
                f"available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB"
            )

        # Загрузка модели выравнивания с progress bar
        model_name = align_model or f"default ({language_code})"
        model_progress = ModelLoadingProgress(model_name, "модели выравнивания")
        model_progress.start()
        model_progress.update(30)

        try:
            align_model_loaded, align_metadata = load_align_model(
                language_code=language_code, device=device, model_name=align_model
            )
        except OSError as e:
            if "No space left on device" in str(e) or "os error 28" in str(e):
                error_msg = (
                    f"Недостаточно места на диске при загрузке модели выравнивания для языка {language_code}. "
                    f"Ошибка: {str(e)}"
                )
                self.logger.error(error_msg)
                raise AudioProcessingError(
                    reason=error_msg,
                    original_error=e,
                ) from e
            raise

        model_progress.update(100)
        model_progress.complete()

        if progress_callback:
            progress_callback.update_step(50)  # Модель загружена - 50% выравнивания

        # Выравнивание
        self.logger.info(f"   🔗 Начало выравнивания (метод: {interpolate_method})")
        result = align(
            transcript,
            align_model_loaded,
            align_metadata,
            audio,
            device,
            interpolate_method=interpolate_method,
            return_char_alignments=return_char_alignments,
        )

        if progress_callback:
            progress_callback.update_step(100)  # Выравнивание завершено

        # Log GPU memory before cleanup
        if torch.cuda.is_available():
            self.logger.debug(
                f"GPU memory before cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, "
                f"available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB"
            )

        # Clean up model
        gc.collect()
        torch.cuda.empty_cache()
        del align_model_loaded
        del align_metadata

        # Log GPU memory after cleanup
        if torch.cuda.is_available():
            self.logger.debug(
                f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, "
                f"available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB"
            )

        return result  # type: ignore[no-any-return]

    def load_model(
        self, language_code: str, device: str, model_name: str | None = None
    ) -> None:
        """
        Load alignment model for a specific language.

        Args:
            language_code: Language code for the alignment model
            device: Device to load model on ('cpu' or 'cuda')
            model_name: Specific model name to use (optional)
        """
        self.logger.info(f"Loading alignment model for {language_code} on {device}")
        self.model, self.metadata = load_align_model(
            language_code=language_code, device=device, model_name=model_name
        )

    def unload_model(self) -> None:
        """Unload alignment model and free GPU memory."""
        if self.model:
            del self.model
            self.model = None
        if self.metadata:
            del self.metadata
            self.metadata = None
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.debug("Alignment model unloaded and GPU memory cleared")
