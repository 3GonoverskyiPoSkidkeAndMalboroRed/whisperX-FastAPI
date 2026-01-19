"""Скрипт для предустановки моделей WhisperX при запуске контейнера."""

import os
import sys
from typing import Any

import logging

import torch
from whisperx import load_align_model, load_model
from whisperx.diarize import DiarizationPipeline

# Настройка логирования для скрипта
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def preload_transcription_models(models: list[str], device: str, compute_type: str) -> None:
    """
    Предзагружает модели транскрипции.
    
    Args:
        models: Список названий моделей для загрузки
        device: Устройство для загрузки ('cpu' или 'cuda')
        compute_type: Тип вычислений ('float16', 'int8', и т.д.)
    """
    if not models:
        logger.info("Модели транскрипции для предзагрузки не указаны")
        return
    
    logger.info(f"📥 Начало предзагрузки моделей транскрипции: {', '.join(models)}")
    
    for model_name in models:
        try:
            logger.info(f"   Загрузка модели транскрипции: {model_name}")
            model = load_model(
                model_name,
                device=device,
                compute_type=compute_type,
            )
            logger.info(f"   ✅ Модель транскрипции {model_name} успешно загружена")
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            logger.error(f"   ❌ Ошибка при загрузке модели {model_name}: {e}")
            # Продолжаем загрузку других моделей даже при ошибке
    
    logger.info("✅ Предзагрузка моделей транскрипции завершена")


def preload_alignment_models(languages: list[str], device: str) -> None:
    """
    Предзагружает модели выравнивания для указанных языков.
    
    Args:
        languages: Список кодов языков для загрузки моделей выравнивания
        device: Устройство для загрузки ('cpu' или 'cuda')
    """
    if not languages:
        logger.info("Языки для предзагрузки моделей выравнивания не указаны")
        return
    
    logger.info(f"📥 Начало предзагрузки моделей выравнивания для языков: {', '.join(languages)}")
    
    for lang_code in languages:
        try:
            logger.info(f"   Загрузка модели выравнивания для языка: {lang_code}")
            align_model, align_metadata = load_align_model(
                language_code=lang_code,
                device=device,
            )
            logger.info(f"   ✅ Модель выравнивания для {lang_code} успешно загружена")
            del align_model, align_metadata
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            logger.error(f"   ❌ Ошибка при загрузке модели выравнивания для {lang_code}: {e}")
            # Продолжаем загрузку других моделей даже при ошибке
    
    logger.info("✅ Предзагрузка моделей выравнивания завершена")


def preload_diarization_model(device: str, hf_token: str | None) -> None:
    """
    Предзагружает модель диаризации.
    
    Args:
        device: Устройство для загрузки ('cpu' или 'cuda')
        hf_token: HuggingFace токен для доступа к модели
    """
    if not hf_token:
        logger.warning("HF_TOKEN не указан, пропускаем предзагрузку модели диаризации")
        return
    
    logger.info("📥 Начало предзагрузки модели диаризации")
    
    try:
        logger.info("   Загрузка модели диаризации: PyAnnote")
        model = DiarizationPipeline(use_auth_token=hf_token, device=device)
        logger.info("   ✅ Модель диаризации успешно загружена")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        logger.error(f"   ❌ Ошибка при загрузке модели диаризации: {e}")
    
    logger.info("✅ Предзагрузка модели диаризации завершена")


def main() -> None:
    """Основная функция для предзагрузки моделей."""
    logger.info("=" * 80)
    logger.info("🚀 Начало предзагрузки моделей WhisperX")
    logger.info("=" * 80)
    
    # Определяем устройство
    device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    
    # Определяем compute_type в зависимости от устройства
    compute_type = os.getenv("COMPUTE_TYPE")
    if not compute_type:
        if device == "cuda":
            compute_type = "float16"
        else:
            compute_type = "int8"
    
    logger.info(f"Устройство: {device}, Compute type: {compute_type}")
    
    # Предзагрузка моделей транскрипции
    transcription_models_str = os.getenv("PRELOAD_TRANSCRIPTION_MODELS", "")
    if transcription_models_str:
        transcription_models = [m.strip() for m in transcription_models_str.split(",") if m.strip()]
        preload_transcription_models(transcription_models, device, compute_type)
    
    # Предзагрузка моделей выравнивания
    alignment_languages_str = os.getenv("PRELOAD_ALIGNMENT_LANGUAGES", "")
    if alignment_languages_str:
        alignment_languages = [l.strip() for l in alignment_languages_str.split(",") if l.strip()]
        preload_alignment_models(alignment_languages, device)
    
    # Предзагрузка модели диаризации
    preload_diarization = os.getenv("PRELOAD_DIARIZATION", "false").lower() == "true"
    if preload_diarization:
        hf_token = os.getenv("HF_TOKEN")
        preload_diarization_model(device, hf_token)
    
    logger.info("=" * 80)
    logger.info("✨ Предзагрузка моделей завершена")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Предзагрузка моделей прервана пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Критическая ошибка при предзагрузке моделей: {e}", exc_info=True)
        sys.exit(1)
