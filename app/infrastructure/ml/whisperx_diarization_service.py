"""WhisperX implementation of diarization service."""

import gc
import threading
from typing import Any

import numpy as np
import pandas as pd
import torch
from whisperx.diarize import DiarizationPipeline

from app.core.logging import logger
from app.utils.progress import ModelLoadingProgress


class WhisperXDiarizationService:
    """
    WhisperX/PyAnnote-based implementation of diarization service.

    This service wraps the WhisperX diarization pipeline (PyAnnote) to provide
    speaker diarization functionality following the IDiarizationService interface.
    
    Models are cached in memory to avoid reloading on each request.
    """

    def __init__(self, hf_token: str) -> None:
        """
        Initialize the diarization service.

        Args:
            hf_token: HuggingFace authentication token for model access
        """
        self.hf_token = hf_token
        self.model: Any = None
        self.logger = logger
        # Кэш моделей: ключ - строка параметров, значение - загруженная модель
        self._model_cache: dict[str, Any] = {}
        # Блокировка для потокобезопасности
        self._cache_lock = threading.Lock()

    def diarize(
        self,
        audio: np.ndarray[Any, np.dtype[np.float32]],
        device: str,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        progress_callback: Any = None,
    ) -> pd.DataFrame:
        """
        Identify speakers using PyAnnote diarization model.

        Args:
            audio: Audio data as numpy array (float32)
            device: Device to use ('cpu' or 'cuda')
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)
            progress_callback: Callback для обновления прогресса (опционально)

        Returns:
            DataFrame with speaker segments
        """
        # Log GPU memory before loading model
        if torch.cuda.is_available():
            self.logger.debug(
                f"GPU memory before loading model - used: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, "
                f"available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB"
            )

        # Создаём ключ кэша на основе параметров модели
        cache_key = self._create_cache_key(device=device)

        # Проверяем кэш и загружаем модель только если её нет
        with self._cache_lock:
            if cache_key in self._model_cache:
                self.logger.info(f"   ✅ Использование кэшированной модели диаризации")
                model = self._model_cache[cache_key]
            else:
                # Загрузка модели диаризации с progress bar
                model_progress = ModelLoadingProgress("PyAnnote", "модели диаризации")
                model_progress.start()
                model_progress.update(30)

                self.logger.info(f"   📥 Загрузка модели диаризации")
                model = DiarizationPipeline(use_auth_token=self.hf_token, device=device)
                
                # Сохраняем модель в кэш
                self._model_cache[cache_key] = model
                self.logger.info(f"   💾 Модель диаризации сохранена в кэш")

                model_progress.update(100)
                model_progress.complete()

        if progress_callback:
            progress_callback.update_step(50)  # Модель загружена - 50% диаризации

        # Диаризация
        speakers_info = f"min={min_speakers}, max={max_speakers}" if min_speakers or max_speakers else "автоопределение"
        self.logger.info(f"   🎭 Начало диаризации (спикеры: {speakers_info})")
        result = model(
            audio=audio, min_speakers=min_speakers, max_speakers=max_speakers
        )

        if progress_callback:
            progress_callback.update_step(100)  # Диаризация завершена

        # НЕ удаляем модель - она остаётся в кэше для повторного использования
        # Log GPU memory after diarization
        if torch.cuda.is_available():
            self.logger.debug(
                f"GPU memory after diarization: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, "
                f"available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB"
            )

        return result  # type: ignore[no-any-return]

    def _create_cache_key(self, device: str) -> str:
        """
        Создаёт ключ кэша на основе параметров модели диаризации.
        
        Args:
            device: Устройство
            
        Returns:
            Строковый ключ для кэша
        """
        # Для диаризации модель зависит только от устройства
        # hf_token уже учтён в __init__, поэтому не включаем его в ключ
        return f"diarization_{device}"

    def load_model(self, device: str, hf_token: str) -> None:
        """
        Load diarization model.

        Args:
            device: Device to load model on ('cpu' or 'cuda')
            hf_token: HuggingFace authentication token
        """
        self.logger.info(f"Loading diarization model on {device}")
        self.hf_token = hf_token
        cache_key = self._create_cache_key(device)
        
        with self._cache_lock:
            if cache_key not in self._model_cache:
                self.model = DiarizationPipeline(use_auth_token=self.hf_token, device=device)
                self._model_cache[cache_key] = self.model
            else:
                self.model = self._model_cache[cache_key]

    def unload_model(self, cache_key: str | None = None) -> None:
        """
        Unload diarization model and free GPU memory.
        
        Args:
            cache_key: Ключ модели для удаления из кэша. Если None, удаляет все модели.
        """
        with self._cache_lock:
            if cache_key:
                if cache_key in self._model_cache:
                    del self._model_cache[cache_key]
                    self.logger.debug(f"Diarization model {cache_key} unloaded from cache")
            else:
                # Удаляем все модели из кэша
                self._model_cache.clear()
                self.logger.debug("All diarization models unloaded from cache")
        
        if self.model:
            del self.model
            self.model = None
        
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.debug("GPU memory cleared")
