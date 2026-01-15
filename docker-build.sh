#!/bin/bash
# Скрипт для сборки и запуска Docker контейнеров с BuildKit
# 
# ПРИМЕЧАНИЕ: BuildKit теперь автоматически включен через ~/.zshrc
# Этот скрипт оставлен для совместимости и дополнительной гарантии
# 
# Использование: ./docker-build.sh [аргументы для docker-compose up]

# Явное включение BuildKit (на случай, если переменные не загружены)
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Остановка и удаление существующих контейнеров
docker-compose down


# Показать статус запущенных контейнеров
docker ps
# Сборка и запуск контейнеров с переданными аргументами
docker-compose up --build 



