"""This module contains the task management routes for the FastAPI application."""

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from app.api.dependencies import get_task_management_service
from app.api.mappers.task_mapper import TaskMapper
from app.api.schemas.task_schemas import TaskListResponse
from app.core.exceptions import TaskNotFoundError
from app.core.logging import logger
from app.schemas import Metadata, Response, Result
from app.services.task_management_service import TaskManagementService
from app.utils.transcript_formatter import format_transcript_to_text

task_router = APIRouter()


@task_router.get("/task/all", tags=["Tasks Management"])
async def get_all_tasks_status(
    service: TaskManagementService = Depends(get_task_management_service),
) -> TaskListResponse:
    """
    Retrieve the status of all tasks.

    Args:
        service: Task management service dependency.

    Returns:
        TaskListResponse: The status of all tasks.
    """
    logger.info("Retrieving status of all tasks")
    tasks = service.get_all_tasks()

    # Convert domain tasks to API DTOs using mapper
    task_summaries = [TaskMapper.to_summary(task) for task in tasks]

    return TaskListResponse(tasks=task_summaries)


@task_router.get("/task/{identifier}", tags=["Tasks Management"])
async def get_transcription_status(
    identifier: str,
    service: TaskManagementService = Depends(get_task_management_service),
) -> Result:
    """
    Retrieve the status of a specific task by its identifier.

    Args:
        identifier (str): The identifier of the task.
        service: Task management service dependency.

    Returns:
        Result: The status of the task.

    Raises:
        TaskNotFoundError: If the identifier is not found.
    """
    logger.info("Retrieving status for task ID: %s", identifier)
    task = service.get_task(identifier)

    if task is None:
        logger.error("Task ID not found: %s", identifier)
        raise TaskNotFoundError(identifier)

    logger.info("Status retrieved for task ID: %s", identifier)
    return Result(
        status=task.status,
        result=task.result,
        metadata=Metadata(
            task_type=task.task_type,
            task_params=task.task_params,
            language=task.language,
            file_name=task.file_name,
            url=task.url,
            callback_url=task.callback_url,
            duration=task.duration,
            audio_duration=task.audio_duration,
            start_time=task.start_time,
            end_time=task.end_time,
        ),
        error=task.error,
    )


@task_router.get("/task/{identifier}/text", tags=["Tasks Management"])
async def get_transcription_text(
    identifier: str,
    service: TaskManagementService = Depends(get_task_management_service),
) -> PlainTextResponse:
    """
    Получить результат транскрипции в текстовом формате.

    Возвращает результат транскрипции в читаемом текстовом формате:
    [SPEAKER_00]: Текст сегмента.
    [SPEAKER_01]: Другой текст.

    Args:
        identifier (str): Идентификатор задачи.
        service: Сервис управления задачами.

    Returns:
        PlainTextResponse: Текст транскрипции в читаемом формате.

    Raises:
        TaskNotFoundError: Если задача не найдена.
    """
    logger.info("Retrieving text format for task ID: %s", identifier)
    task = service.get_task(identifier)

    if task is None:
        logger.error("Task ID not found: %s", identifier)
        raise TaskNotFoundError(identifier)

    if task.status != "completed":
        logger.warning("Task %s is not completed, status: %s", identifier, task.status)
        return PlainTextResponse(
            content=f"Задача еще не завершена. Статус: {task.status}",
            status_code=200,
        )

    formatted_text = format_transcript_to_text(task.result)
    logger.info("Text format retrieved for task ID: %s", identifier)
    return PlainTextResponse(content=formatted_text, media_type="text/plain; charset=utf-8")


@task_router.delete("/task/{identifier}/delete", tags=["Tasks Management"])
async def delete_task(
    identifier: str,
    service: TaskManagementService = Depends(get_task_management_service),
) -> Response:
    """
    Delete a specific task by its identifier.

    Args:
        identifier (str): The identifier of the task.
        service: Task management service dependency.

    Returns:
        Response: Confirmation message of task deletion.

    Raises:
        TaskNotFoundError: If the task is not found.
    """
    logger.info("Deleting task ID: %s", identifier)
    if service.delete_task(identifier):
        logger.info("Task deleted: ID %s", identifier)
        return Response(identifier=identifier, message="Task deleted")
    else:
        logger.error("Task not found: ID %s", identifier)
        raise TaskNotFoundError(identifier)
