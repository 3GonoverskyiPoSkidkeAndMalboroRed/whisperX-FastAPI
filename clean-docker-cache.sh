#!/bin/bash
# Скрипт для очистки кэша Docker при нехватке места на диске

echo "Очистка кэша Docker..."

# Остановка всех контейнеров
echo "Остановка контейнеров..."
docker stop $(docker ps -aq) 2>/dev/null || true

# Удаление неиспользуемых образов
echo "Удаление неиспользуемых образов..."
docker image prune -a -f

# Удаление неиспользуемых контейнеров
echo "Удаление неиспользуемых контейнеров..."
docker container prune -f

# Удаление неиспользуемых volumes
echo "Удаление неиспользуемых volumes..."
docker volume prune -f

# Удаление неиспользуемых сетей
echo "Удаление неиспользуемых сетей..."
docker network prune -f

# Очистка build cache
echo "Очистка build cache..."
docker builder prune -a -f

# Полная очистка системы (осторожно!)
echo "Полная очистка системы Docker..."
docker system prune -a -f --volumes

echo "Очистка завершена!"
echo "Освобождено место. Теперь можно пересобрать образ:"
echo "  docker-compose build --no-cache"
