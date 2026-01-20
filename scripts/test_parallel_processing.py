"""Тестовый скрипт для проверки параллельной обработки задач на GPU.

Этот скрипт отправляет несколько запросов одновременно на эндпоинт POST /speech-to-text
для проверки работы очереди задач и параллелизма обработки.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import httpx


# Добавляем корневую директорию проекта в путь для импортов
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ParallelTestClient:
    """Клиент для тестирования параллельной обработки."""

    def __init__(
        self,
        base_url: str = "http://localhost:8123",
        audio_file: str = "Ibrashin.mp3",
        num_requests: int = 5,
        concurrent_limit: int = 5,
    ):
        """
        Инициализация тестового клиента.

        Args:
            base_url: Базовый URL API сервера
            audio_file: Путь к аудио файлу для отправки
            num_requests: Количество запросов для отправки
            concurrent_limit: Максимальное количество одновременных запросов
        """
        self.base_url = base_url.rstrip("/")
        self.audio_file = Path(audio_file)
        self.num_requests = num_requests
        self.concurrent_limit = concurrent_limit
        self.results: list[dict[str, Any]] = []

        if not self.audio_file.exists():
            raise FileNotFoundError(
                f"Аудио файл не найден: {self.audio_file.absolute()}"
            )

    async def send_request(
        self, client: httpx.AsyncClient, request_id: int
    ) -> dict[str, Any]:
        """
        Отправить один запрос на обработку аудио.

        Args:
            client: HTTP клиент
            request_id: ID запроса для отслеживания

        Returns:
            Словарь с результатами запроса
        """
        start_time = time.time()
        try:
            # Открываем файл для отправки
            with open(self.audio_file, "rb") as f:
                files = {"file": (self.audio_file.name, f, "audio/mpeg")}
                params = {
                    "model": "tiny",  # Используем маленькую модель для быстрого теста
                    "language": "ru",
                    "device": "cuda",
                }

                print(f"[Запрос {request_id}] Отправка файла {self.audio_file.name}...")

                response = await client.post(
                    f"{self.base_url}/speech-to-text",
                    files=files,
                    params=params,
                    timeout=300.0,  # 5 минут таймаут
                )

                response.raise_for_status()
                result = response.json()

                elapsed_time = time.time() - start_time

                print(
                    f"[Запрос {request_id}] ✅ Успешно! "
                    f"Task ID: {result.get('identifier')}, "
                    f"Время ответа: {elapsed_time:.2f}с"
                )

                return {
                    "request_id": request_id,
                    "status": "success",
                    "task_id": result.get("identifier"),
                    "response_time": elapsed_time,
                    "message": result.get("message"),
                    "error": None,
                }

        except httpx.HTTPStatusError as e:
            elapsed_time = time.time() - start_time
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            print(f"[Запрос {request_id}] ❌ Ошибка HTTP: {error_msg}")

            return {
                "request_id": request_id,
                "status": "error",
                "task_id": None,
                "response_time": elapsed_time,
                "message": None,
                "error": error_msg,
            }

        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = str(e)
            print(f"[Запрос {request_id}] ❌ Ошибка: {error_msg}")

            return {
                "request_id": request_id,
                "status": "error",
                "task_id": None,
                "response_time": elapsed_time,
                "message": None,
                "error": error_msg,
            }

    async def get_queue_stats(self, client: httpx.AsyncClient) -> dict[str, Any]:
        """
        Получить статистику очереди задач.

        Args:
            client: HTTP клиент

        Returns:
            Словарь со статистикой очереди
        """
        try:
            response = await client.get(f"{self.base_url}/queue/stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"⚠️  Не удалось получить статистику очереди: {e}")
            return {}

    async def run_test(self) -> None:
        """Запустить тест параллельной обработки."""
        print("=" * 80)
        print("🚀 ТЕСТ ПАРАЛЛЕЛЬНОЙ ОБРАБОТКИ ЗАДАЧ")
        print("=" * 80)
        print(f"📁 Файл: {self.audio_file.absolute()}")
        print(f"🌐 URL: {self.base_url}")
        print(f"📊 Количество запросов: {self.num_requests}")
        print(f"⚡ Одновременных запросов: {self.concurrent_limit}")
        print("=" * 80)
        print()

        # Получаем начальную статистику
        async with httpx.AsyncClient() as client:
            initial_stats = await self.get_queue_stats(client)
            if initial_stats:
                print("📈 Начальная статистика очереди:")
                print(f"   Размер очереди: {initial_stats.get('queue_size', 0)}")
                print(
                    f"   Всего отправлено: {initial_stats.get('total_submitted', 0)}"
                )
                print(
                    f"   Всего обработано: {initial_stats.get('total_processed', 0)}"
                )
                print(
                    f"   Макс. параллельных задач: {initial_stats.get('max_concurrent_tasks', 'N/A')}"
                )
                print()

        # Засекаем общее время
        test_start_time = time.time()

        # Создаем семафор для ограничения одновременных запросов
        semaphore = asyncio.Semaphore(self.concurrent_limit)

        async def send_with_semaphore(request_id: int) -> dict[str, Any]:
            """Отправить запрос с ограничением через семафор."""
            async with semaphore:
                async with httpx.AsyncClient() as client:
                    return await self.send_request(client, request_id)

        # Создаем задачи для всех запросов
        tasks = [
            send_with_semaphore(i + 1) for i in range(self.num_requests)
        ]

        print(f"⏳ Отправка {self.num_requests} запросов...")
        print()

        # Выполняем все запросы параллельно
        results = await asyncio.gather(*tasks)

        test_elapsed_time = time.time() - test_start_time

        # Сохраняем результаты
        self.results = results

        print()
        print("=" * 80)
        print("📊 РЕЗУЛЬТАТЫ ТЕСТА")
        print("=" * 80)

        # Статистика по результатам
        successful = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "error")
        avg_response_time = (
            sum(r["response_time"] for r in results) / len(results)
            if results
            else 0
        )

        print(f"✅ Успешных запросов: {successful}/{self.num_requests}")
        print(f"❌ Неудачных запросов: {failed}/{self.num_requests}")
        print(f"⏱️  Среднее время ответа: {avg_response_time:.2f}с")
        print(f"⏱️  Общее время теста: {test_elapsed_time:.2f}с")
        print()

        # Показываем детали по каждому запросу
        print("📋 Детали запросов:")
        for result in results:
            status_icon = "✅" if result["status"] == "success" else "❌"
            print(
                f"   {status_icon} Запрос {result['request_id']}: "
                f"{result['status']} "
                f"(Task ID: {result.get('task_id', 'N/A')}, "
                f"Время: {result['response_time']:.2f}с)"
            )
            if result["error"]:
                print(f"      Ошибка: {result['error']}")

        print()

        # Получаем финальную статистику
        async with httpx.AsyncClient() as client:
            final_stats = await self.get_queue_stats(client)
            if final_stats:
                print("📈 Финальная статистика очереди:")
                print(f"   Размер очереди: {final_stats.get('queue_size', 0)}")
                print(
                    f"   Всего отправлено: {final_stats.get('total_submitted', 0)}"
                )
                print(
                    f"   Всего обработано: {final_stats.get('total_processed', 0)}"
                )
                print(
                    f"   Всего ошибок: {final_stats.get('total_failed', 0)}"
                )
                print(
                    f"   Активных воркеров: {final_stats.get('active_workers', 'N/A')}"
                )

        print("=" * 80)

        # Анализ параллелизма
        if successful > 0:
            print()
            print("🔍 АНАЛИЗ ПАРАЛЛЕЛЛИЗМА:")
            print("=" * 80)

            # Если все запросы были успешными и отправлены быстро,
            # значит они попали в очередь и обрабатываются параллельно
            if avg_response_time < 5.0:  # Быстрый ответ означает, что задача добавлена в очередь
                print(
                    "✅ Запросы успешно добавлены в очередь (быстрый ответ сервера)"
                )
                print(
                    "✅ Параллельная обработка должна работать, проверьте логи сервера"
                )
                print(
                    "   и статистику задач для подтверждения параллельного выполнения"
                )
            else:
                print(
                    "⚠️  Время ответа сервера большое, возможно задачи обрабатываются последовательно"
                )

            print("=" * 80)


async def main() -> None:
    """Главная функция."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Тест параллельной обработки задач на GPU"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8123",
        help="URL API сервера (по умолчанию: http://localhost:8123)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="Ibrashin.mp3",
        help="Путь к аудио файлу (по умолчанию: Ibrashin.mp3)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=5,
        help="Количество запросов для отправки (по умолчанию: 5)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=5,
        help="Максимальное количество одновременных запросов (по умолчанию: 5)",
    )

    args = parser.parse_args()

    # Проверяем существование файла
    audio_file = Path(args.file)
    if not audio_file.is_absolute():
        # Пробуем найти файл в корне проекта
        project_root = Path(__file__).parent.parent
        audio_file = project_root / args.file

    if not audio_file.exists():
        print(f"❌ Ошибка: Файл не найден: {audio_file.absolute()}")
        print(f"   Проверьте путь к файлу или используйте --file для указания пути")
        sys.exit(1)

    client = ParallelTestClient(
        base_url=args.url,
        audio_file=str(audio_file),
        num_requests=args.num_requests,
        concurrent_limit=args.concurrent,
    )

    try:
        await client.run_test()
    except KeyboardInterrupt:
        print("\n\n⚠️  Тест прерван пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
