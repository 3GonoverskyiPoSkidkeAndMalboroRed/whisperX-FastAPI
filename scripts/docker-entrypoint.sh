#!/bin/bash
# Не используем set -e, чтобы приложение запускалось даже при ошибке предзагрузки

# Функция для логирования
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

# Проверяем, нужно ли предзагружать модели
PRELOAD_TRANSCRIPTION_MODELS="${PRELOAD_TRANSCRIPTION_MODELS:-}"
PRELOAD_ALIGNMENT_LANGUAGES="${PRELOAD_ALIGNMENT_LANGUAGES:-}"
PRELOAD_DIARIZATION="${PRELOAD_DIARIZATION:-false}"

# Если указаны модели для предзагрузки, запускаем скрипт
if [ -n "$PRELOAD_TRANSCRIPTION_MODELS" ] || [ -n "$PRELOAD_ALIGNMENT_LANGUAGES" ] || [ "$PRELOAD_DIARIZATION" = "true" ]; then
    log "Запуск предзагрузки моделей..."
    if ! uv run python scripts/preload_models.py; then
        log "Предупреждение: предзагрузка моделей завершилась с ошибками, но продолжаем запуск приложения"
    fi
else
    log "Предзагрузка моделей пропущена (не указаны переменные окружения)"
fi

# Запускаем основное приложение
log "Запуск приложения..."
exec uv run gunicorn \
    --bind 0.0.0.0:8000 \
    --workers 1 \
    --timeout 0 \
    --log-config gunicorn_logging.conf \
    app.main:app \
    -k uvicorn.workers.UvicornWorker
