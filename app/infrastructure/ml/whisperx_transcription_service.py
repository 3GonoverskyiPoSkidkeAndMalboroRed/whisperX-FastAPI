"""WhisperX implementation of transcription service."""

import gc
from typing import Any

import numpy as np
import torch
from whisperx import load_model

from app.core.logging import logger
from app.utils.progress import ModelLoadingProgress


class WhisperXTranscriptionService:
    """
    WhisperX-based implementation of transcription service.

    This service wraps the WhisperX library to provide transcription
    functionality following the ITranscriptionService interface contract.
    """

    def __init__(self) -> None:
        """Initialize the transcription service."""
        self.model: Any = None
        self.logger = logger

    def transcribe(
        self,
        audio: np.ndarray[Any, np.dtype[np.float32]],
        task: str,
        asr_options: dict[str, Any],
        vad_options: dict[str, Any],
        language: str,
        batch_size: int,
        chunk_size: int,
        model: str,
        device: str,
        device_index: int,
        compute_type: str,
        threads: int,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """
        Transcribe audio using WhisperX model.

        Args:
            audio: Audio data as numpy array (float32)
            task: Transcription task type ('transcribe' or 'translate')
            asr_options: ASR model options
            vad_options: Voice Activity Detection options
            language: Language code for transcription
            batch_size: Batch size for processing
            chunk_size: Chunk size for processing
            model: Model name/size to use
            device: Device to use ('cpu' or 'cuda')
            device_index: Device index for multi-GPU setups
            compute_type: Computation precision ('float16', 'int8', etc.)
            threads: Number of threads to use
            progress_callback: Callback для обновления прогресса (опционально)

        Returns:
            Dictionary containing transcription results
        """
        # Log GPU memory before loading model
        if torch.cuda.is_available():
            gpu_memory_before = torch.cuda.memory_allocated() / 1024**2
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            self.logger.debug(
                f"GPU memory before loading model - used: {gpu_memory_before:.2f} MB, "
                f"available: {gpu_memory_total:.2f} MB"
            )

        # Set thread count
        faster_whisper_threads = 4
        if threads > 0:
            torch.set_num_threads(threads)
            faster_whisper_threads = threads

        # Загрузка модели с progress bar
        model_progress = ModelLoadingProgress(model, "модели транскрипции")
        model_progress.start()
        model_progress.update(30)  # Начало загрузки

        loaded_model = load_model(
            model,
            device,
            device_index=device_index,
            compute_type=compute_type,
            asr_options=asr_options,
            vad_options=vad_options,
            language=language,
            task=task,
            threads=faster_whisper_threads,
        )

        model_progress.update(100)
        model_progress.complete()

        if progress_callback:
            progress_callback.update_step(50)  # Модель загружена - 50% транскрипции

        # Транскрипция
        self.logger.info(f"   🎤 Начало транскрипции (batch_size={batch_size}, chunk_size={chunk_size})")
        result = loaded_model.transcribe(
            audio=audio, batch_size=batch_size, chunk_size=chunk_size, language=language
        )

        if progress_callback:
            progress_callback.update_step(100)  # Транскрипция завершена

        # Log GPU memory before cleanup
        if torch.cuda.is_available():
            self.logger.debug(
                f"GPU memory before cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, "
                f"available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB"
            )

        # Clean up model
        gc.collect()
        torch.cuda.empty_cache()
        del loaded_model

        # Log GPU memory after cleanup
        if torch.cuda.is_available():
            self.logger.debug(
                f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, "
                f"available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB"
            )

        return result  # type: ignore[no-any-return]

    def load_model(
        self,
        model_name: str,
        device: str,
        device_index: int,
        compute_type: str,
        asr_options: dict[str, Any],
        vad_options: dict[str, Any],
        language: str,
        task: str,
        threads: int,
    ) -> None:
        """
        Load WhisperX model.

        Args:
            model_name: Name/size of the model to load
            device: Device to load model on ('cpu' or 'cuda')
            device_index: Device index for multi-GPU setups
            compute_type: Computation precision
            asr_options: ASR model options
            vad_options: Voice Activity Detection options
            language: Target language
            task: Task type
            threads: Number of threads to use
        """
        self.logger.info(f"Loading model {model_name} on {device}")

        faster_whisper_threads = 4
        if threads > 0:
            torch.set_num_threads(threads)
            faster_whisper_threads = threads

        self.model = load_model(
            model_name,
            device,
            device_index=device_index,
            compute_type=compute_type,
            asr_options=asr_options,
            vad_options=vad_options,
            language=language,
            task=task,
            threads=faster_whisper_threads,
        )

    def unload_model(self) -> None:
        """Unload WhisperX model and free GPU memory."""
        if self.model:
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
            self.logger.debug("Model unloaded and GPU memory cleared")
