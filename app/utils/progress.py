"""Утилиты для красивого вывода прогресса транскрипции."""

import sys
from typing import Any

from tqdm import tqdm

from app.core.logging import logger


class TranscriptionProgress:
    """Класс для управления прогрессом транскрипции с красивым выводом."""

    def __init__(self, identifier: str, total_steps: int = 1) -> None:
        """
        Инициализировать менеджер прогресса.

        Args:
            identifier: Идентификатор задачи
            total_steps: Общее количество этапов
        """
        self.identifier = identifier
        self.total_steps = total_steps
        self.current_step = 0
        self.main_bar: tqdm | None = None
        self.step_bar: tqdm | None = None

    def start(self) -> None:
        """Начать отслеживание прогресса."""
        logger.info("=" * 80)
        logger.info(f"🚀 Начало транскрипции [ID: {self.identifier}]")
        logger.info("=" * 80)
        self.main_bar = tqdm(
            total=self.total_steps,
            desc="📊 Общий прогресс",
            unit="этап",
            file=sys.stdout,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            colour="green",
            ncols=100,
            disable=False,
        )

    def start_step(self, step_name: str, description: str = "") -> None:
        """
        Начать новый этап.

        Args:
            step_name: Название этапа
            description: Описание этапа
        """
        if self.main_bar:
            self.current_step += 1
            self.main_bar.update(1)
            self.main_bar.set_description(f"📊 Общий прогресс [{self.current_step}/{self.total_steps}]")

        logger.info("")
        logger.info(f"▶️  Этап {self.current_step}/{self.total_steps}: {step_name}")
        if description:
            logger.info(f"   {description}")
        logger.info("-" * 80)

        self.step_bar = tqdm(
            total=100,
            desc=f"   ⏳ {step_name}",
            unit="%",
            file=sys.stdout,
            bar_format="{l_bar}{bar}| {n_fmt}% [{elapsed}]",
            colour="cyan",
            leave=False,
            ncols=100,
            disable=False,
        )

    def update_step(self, value: int) -> None:
        """
        Обновить прогресс текущего этапа.

        Args:
            value: Значение прогресса (0-100)
        """
        if self.step_bar:
            # Ограничиваем значение от 0 до 100
            value = max(0, min(100, value))
            current = int(self.step_bar.n)
            if value > current:
                self.step_bar.update(value - current)

    def complete_step(self, step_name: str, message: str = "") -> None:
        """
        Завершить текущий этап.

        Args:
            step_name: Название этапа
            message: Дополнительное сообщение
        """
        if self.step_bar:
            self.step_bar.update(100 - self.step_bar.n)
            self.step_bar.close()
            self.step_bar = None

        logger.info(f"✅ Этап завершен: {step_name}")
        if message:
            logger.info(f"   {message}")

    def finish(self, duration: float | None = None) -> None:
        """
        Завершить отслеживание прогресса.

        Args:
            duration: Длительность процесса в секундах
        """
        if self.main_bar:
            self.main_bar.close()
            self.main_bar = None

        logger.info("")
        logger.info("=" * 80)
        if duration:
            logger.info(f"✨ Транскрипция завершена [ID: {self.identifier}] за {duration:.2f}с")
        else:
            logger.info(f"✨ Транскрипция завершена [ID: {self.identifier}]")
        logger.info("=" * 80)
        logger.info("")

    def error(self, error_message: str) -> None:
        """
        Зафиксировать ошибку.

        Args:
            error_message: Сообщение об ошибке
        """
        if self.step_bar:
            self.step_bar.close()
            self.step_bar = None

        if self.main_bar:
            self.main_bar.close()
            self.main_bar = None

        logger.error("")
        logger.error("=" * 80)
        logger.error(f"❌ Ошибка транскрипции [ID: {self.identifier}]")
        logger.error(f"   {error_message}")
        logger.error("=" * 80)
        logger.error("")


class ModelLoadingProgress:
    """Класс для отслеживания загрузки модели."""

    def __init__(self, model_name: str, model_type: str = "модель") -> None:
        """
        Инициализировать отслеживание загрузки модели.

        Args:
            model_name: Название модели
            model_type: Тип модели (модель, модель выравнивания, модель диаризации)
        """
        self.model_name = model_name
        self.model_type = model_type
        self.bar: tqdm | None = None

    def start(self) -> None:
        """Начать отслеживание загрузки."""
        logger.info(f"   📥 Загрузка {self.model_type}: {self.model_name}")
        self.bar = tqdm(
            total=100,
            desc=f"      Загрузка {self.model_type}",
            unit="%",
            file=sys.stdout,
            bar_format="{l_bar}{bar}| {n_fmt}% [{elapsed}]",
            colour="yellow",
            leave=False,
            ncols=100,
            disable=False,
        )

    def update(self, value: int) -> None:
        """
        Обновить прогресс загрузки.

        Args:
            value: Значение прогресса (0-100)
        """
        if self.bar:
            self.bar.update(value - self.bar.n)

    def complete(self) -> None:
        """Завершить отслеживание загрузки."""
        if self.bar:
            self.bar.update(100 - self.bar.n)
            self.bar.close()
            self.bar = None
        logger.info(f"   ✅ {self.model_type.capitalize()} загружена: {self.model_name}")


def create_simple_progress(step_name: str) -> tqdm:
    """
    Создать простой progress bar для одного этапа.

    Args:
        step_name: Название этапа

    Returns:
        Объект tqdm progress bar
    """
    logger.info(f"▶️  {step_name}")
    return tqdm(
        total=100,
        desc=f"   ⏳ {step_name}",
        unit="%",
        file=sys.stdout,
        bar_format="{l_bar}{bar}| {n_fmt}% [{elapsed}]",
        colour="cyan",
        ncols=100,
        disable=False,
    )
