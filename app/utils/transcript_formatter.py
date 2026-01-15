"""Утилита для форматирования результатов транскрипции в читаемый текст."""

from typing import Any


def format_transcript_to_text(result: dict[str, Any] | None) -> str:
    """
    Форматирует результат транскрипции в читаемый текстовый формат.

    Преобразует JSON результат с сегментами и спикерами в формат:
    [SPEAKER_00]: Текст сегмента.
    [SPEAKER_01]: Другой текст.

    Args:
        result: Словарь с результатом транскрипции, содержащий поле 'segments'

    Returns:
        Отформатированный текст транскрипции
    """
    if not result:
        return ""

    if not isinstance(result, dict):
        return ""

    segments = result.get("segments", [])
    if not segments:
        return ""

    if not isinstance(segments, list):
        return ""

    formatted_lines = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue

        speaker = segment.get("speaker")
        text = segment.get("text", "")

        if text is None:
            text = ""
        else:
            text = str(text).strip()

        if not text:
            continue

        if speaker:
            formatted_lines.append(f"[{speaker}]: {text}")
        else:
            # Если спикер не указан, просто добавляем текст
            formatted_lines.append(text)

    return "\n".join(formatted_lines)
