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
        
        # Проверка результатов диаризации перед назначением спикеров
        if isinstance(diarization_segments, pd.DataFrame):
            if 'speaker' not in diarization_segments.columns:
                self.logger.error("   ❌ В результатах диаризации отсутствует колонка 'speaker'")
                raise ValueError("Diarization segments missing 'speaker' column")
            
            num_speakers = len(diarization_segments['speaker'].unique())
            self.logger.info(f"   📊 Диаризация содержит {num_speakers} спикеров для назначения")
            
            if num_speakers == 1:
                speaker_name = diarization_segments['speaker'].unique()[0]
                self.logger.warning(
                    f"   ⚠️  Диаризация обнаружила только одного спикера ({speaker_name}). "
                    f"Все слова будут назначены этому спикеру."
                )
            
            # Логируем информацию о временных метках
            if len(diarization_segments) > 0:
                min_time = diarization_segments['start'].min()
                max_time = diarization_segments['end'].max()
                self.logger.debug(
                    f"   📍 Временной диапазон диаризации: {min_time:.2f}s - {max_time:.2f}s "
                    f"({len(diarization_segments)} сегментов)"
                )
        
        # Проверка транскрипции
        if 'segments' not in transcript:
            self.logger.error("   ❌ В транскрипции отсутствует ключ 'segments'")
            raise ValueError("Transcript missing 'segments' key")
        
        transcript_segments = transcript.get('segments', [])
        if len(transcript_segments) == 0:
            self.logger.warning("   ⚠️  Транскрипция не содержит сегментов")
        
        self.logger.info(f"   📝 Транскрипция содержит {len(transcript_segments)} сегментов")
        
        # Проверка временных меток транскрипции
        if len(transcript_segments) > 0:
            transcript_times = []
            for seg in transcript_segments:
                if 'start' in seg and 'end' in seg:
                    transcript_times.append((seg['start'], seg['end']))
            
            if transcript_times:
                min_transcript_time = min(t[0] for t in transcript_times)
                max_transcript_time = max(t[1] for t in transcript_times)
                self.logger.debug(
                    f"   📍 Временной диапазон транскрипции: {min_transcript_time:.2f}s - {max_transcript_time:.2f}s"
                )
                
                # Сравнение временных диапазонов
                if isinstance(diarization_segments, pd.DataFrame) and len(diarization_segments) > 0:
                    diarization_min = diarization_segments['start'].min()
                    diarization_max = diarization_segments['end'].max()
                    time_overlap = min(max_transcript_time, diarization_max) - max(min_transcript_time, diarization_min)
                    if time_overlap <= 0:
                        self.logger.warning(
                            f"   ⚠️  Временные диапазоны диаризации и транскрипции не пересекаются! "
                            f"Это может привести к неправильному назначению спикеров."
                        )

        try:
            result = whisperx.assign_word_speakers(diarization_segments, transcript)
        except Exception as e:
            self.logger.error(f"   ❌ Ошибка при назначении спикеров: {str(e)}")
            self.logger.error(f"   Тип ошибки: {type(e).__name__}")
            raise
        
        # Проверка результата назначения спикеров
        if isinstance(result, dict) and 'segments' in result:
            assigned_segments = result['segments']
            if len(assigned_segments) > 0:
                # Подсчитываем количество уникальных спикеров в результате
                speakers_in_result = set()
                for segment in assigned_segments:
                    if 'speaker' in segment:
                        speakers_in_result.add(segment['speaker'])
                    if 'words' in segment:
                        for word in segment['words']:
                            if 'speaker' in word:
                                speakers_in_result.add(word['speaker'])
                
                num_assigned_speakers = len(speakers_in_result)
                self.logger.info(
                    f"   ✅ Назначение спикеров завершено. "
                    f"В результате {num_assigned_speakers} уникальных спикеров: {sorted(speakers_in_result)}"
                )
                
                if num_assigned_speakers == 1 and num_speakers > 1:
                    self.logger.warning(
                        f"   ⚠️  Проблема: диаризация обнаружила {num_speakers} спикеров, "
                        f"но в результате назначения остался только 1 спикер. "
                        f"Возможно, проблема с сопоставлением временных меток."
                    )

        if progress_callback:
            progress_callback.update_step(100)

        return result  # type: ignore[no-any-return]
