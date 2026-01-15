#!/bin/bash
# Скрипт для создания директорий для volumes Docker

echo "Создание директорий для Docker volumes..."

# Создаем директорию для кэша моделей
if [ ! -d "/data/whisperx/cache" ]; then
    echo "Создание директории /data/whisperx/cache..."
    sudo mkdir -p /data/whisperx/cache
    sudo chmod 755 /data/whisperx/cache
    echo "Директория /data/whisperx/cache создана"
else
    echo "Директория /data/whisperx/cache уже существует"
fi

echo ""
echo "Готово! Теперь можно запустить контейнер:"
echo "  docker-compose up"
