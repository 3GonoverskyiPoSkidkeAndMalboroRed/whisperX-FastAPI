"""Task queue manager for parallel GPU processing."""

import os
import threading
from collections.abc import Callable
from queue import Empty, Queue
from typing import Any

from app.core.logging import logger


class TaskQueueManager:
    """
    Менеджер очереди задач для параллельной обработки на одной GPU.
    
    Использует пул потоков для обработки задач с ограничением
    количества одновременных задач через семафор.
    """

    def __init__(
        self,
        max_concurrent_tasks: int | None = None,
        num_worker_threads: int | None = None,
    ):
        """
        Инициализация менеджера очереди.

        Args:
            max_concurrent_tasks: Максимальное количество одновременных задач на GPU.
                                 Если None, определяется из переменной окружения или по умолчанию 2.
            num_worker_threads: Количество потоков-воркеров для обработки.
                               Если None, определяется из переменной окружения или по умолчанию 4.
        """
        # Определяем параметры из переменных окружения или используем значения по умолчанию
        self.max_concurrent_tasks = (
            max_concurrent_tasks
            or int(os.getenv("MAX_CONCURRENT_GPU_TASKS", "2"))
        )
        self.num_worker_threads = (
            num_worker_threads or int(os.getenv("TASK_WORKER_THREADS", "4"))
        )

        # Очередь задач
        self.task_queue: Queue = Queue()

        # Семафор для ограничения параллелизма на GPU
        self.gpu_semaphore = threading.Semaphore(self.max_concurrent_tasks)

        # Флаг для остановки воркеров
        self._stop_event = threading.Event()

        # Пул потоков-воркеров
        self.worker_threads: list[threading.Thread] = []

        # Статистика
        self.stats = {
            "total_submitted": 0,
            "total_processed": 0,
            "total_failed": 0,
            "queue_size": 0,
        }
        self.stats_lock = threading.Lock()

    def start(self) -> None:
        """Запуск пула воркеров."""
        logger.info(
            f"Запуск TaskQueueManager: {self.num_worker_threads} воркеров, "
            f"макс. параллельных задач: {self.max_concurrent_tasks}"
        )

        for i in range(self.num_worker_threads):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"TaskWorker-{i}",
                daemon=True,
            )
            worker.start()
            self.worker_threads.append(worker)

        logger.info("TaskQueueManager запущен")

    def stop(self, timeout: float = 30.0) -> None:
        """Остановка пула воркеров."""
        logger.info("Остановка TaskQueueManager...")
        self._stop_event.set()

        # Ждем завершения всех воркеров
        for worker in self.worker_threads:
            worker.join(timeout=timeout)

        logger.info("TaskQueueManager остановлен")

    def submit_task(
        self,
        task_func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Добавить задачу в очередь.

        Args:
            task_func: Функция для выполнения задачи
            *args: Позиционные аргументы для функции
            **kwargs: Именованные аргументы для функции
        """
        task = {
            "func": task_func,
            "args": args,
            "kwargs": kwargs,
        }

        self.task_queue.put(task)

        with self.stats_lock:
            self.stats["total_submitted"] += 1
            self.stats["queue_size"] = self.task_queue.qsize()

        logger.debug(
            f"Задача добавлена в очередь. Размер очереди: {self.task_queue.qsize()}"
        )

    def _worker_loop(self) -> None:
        """Основной цикл воркера."""
        thread_name = threading.current_thread().name
        logger.debug(f"{thread_name} запущен")

        while not self._stop_event.is_set():
            try:
                # Получаем задачу из очереди с таймаутом
                try:
                    task = self.task_queue.get(timeout=1.0)
                except Empty:
                    continue

                # Получаем семафор для доступа к GPU
                logger.debug(f"{thread_name} получил задачу, ожидание GPU...")
                self.gpu_semaphore.acquire()

                try:
                    logger.debug(f"{thread_name} начал обработку задачи")

                    # Выполняем задачу
                    task["func"](*task["args"], **task["kwargs"])

                    with self.stats_lock:
                        self.stats["total_processed"] += 1
                        self.stats["queue_size"] = self.task_queue.qsize()

                    logger.debug(f"{thread_name} завершил обработку задачи")

                except Exception as e:
                    logger.error(
                        f"{thread_name} ошибка при обработке задачи: {e}",
                        exc_info=True,
                    )

                    with self.stats_lock:
                        self.stats["total_failed"] += 1
                        self.stats["queue_size"] = self.task_queue.qsize()

                finally:
                    # Освобождаем семафор
                    self.gpu_semaphore.release()
                    self.task_queue.task_done()

            except Exception as e:
                logger.error(
                    f"{thread_name} критическая ошибка: {e}", exc_info=True
                )

        logger.debug(f"{thread_name} остановлен")

    def get_stats(self) -> dict[str, Any]:
        """Получить статистику обработки."""
        with self.stats_lock:
            return self.stats.copy()
