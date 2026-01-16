# -*- coding: utf-8 -*-
# Time      :2024/11/17 15:33
# Author    :Hui Huang
import asyncio
import uuid
from typing import Callable, List, Any, Awaitable, Tuple
from asyncio import Queue


class BatchProcessor:
    """Batch Processor for handling asynchronous requests in batches.

    This class manages a queue of requests and processes them in batches
    using multiple worker tasks.

    Attributes:
        processing_function (Callable[[List[Any]], Awaitable[List[Any]]]):
            The function used for processing requests in batches.
        num_workers (int): The number of worker tasks to process requests.
        batch_size (int): The maximum number of requests to process in a single batch.
        request_queue (Queue): The queue holding incoming requests.
        loop (asyncio.AbstractEventLoop): The event loop used to create worker tasks.
        worker_tasks (List[asyncio.Task]): The list of worker tasks.
    """

    def __init__(
            self,
            processing_function: Callable[[List[Any]], Awaitable[List[Any]]],
            num_workers: int,
            batch_size: int,
            wait_timeout: float = 0.05
    ) -> None:
        """Initialize the BatchProcessor with the given processing function, number of workers, and batch size.

        Args:
            processing_function (Callable[[List[Any]], Awaitable[List[Any]]]):
                The function used for processing requests in batches.
            num_workers (int): The number of worker tasks to process requests.
            batch_size (int): The maximum number of requests to process in a single batch.
        """
        self.processing_function = processing_function
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.wait_timeout = wait_timeout
        self.request_queue: Queue = Queue()
        self.loop = asyncio.get_running_loop()
        self.worker_tasks = [
            self.loop.create_task(self.batch_processor(i)) for i in range(num_workers)
        ]
        # Wait until all worker tasks are started
        self.loop.create_task(self._log_workers_started())

    async def _log_workers_started(self):
        await asyncio.sleep(0)  # Yield control to ensure workers have started

    async def batch_processor(self, worker_id: int):
        """Worker task that processes requests from the queue in batches.

        Args:
            worker_id (int): The identifier for the worker task.
        """

        while True:
            requests: List[Tuple[Any, asyncio.Future]] = []
            try:
                while len(requests) < self.batch_size:
                    request = await asyncio.wait_for(
                        self.request_queue.get(), timeout=self.wait_timeout
                    )
                    requests.append(request)
            except asyncio.TimeoutError:
                pass

            if requests:
                all_requests = [
                    req[0] for req in requests
                ]  # Extract the actual input data from each request tuple
                futures = [req[1] for req in requests]  # Extract the futures to resolve
                try:
                    results = await self.processing_function(all_requests)

                    for (future, result) in zip(futures, results):
                        future.set_result(result)
                except Exception as e:
                    for future in futures:
                        future.set_exception(e)

    async def add_request(self, single_input: Any):
        """Add a new request to the queue.

        Args:
            single_input (Any): The input data for processing.
        """
        # loop = asyncio.get_running_loop()
        future = self.loop.create_future()
        self.request_queue.put_nowait((single_input, future))
        return future

    async def shutdown(self):
        """Shutdown the batch processor by cancelling all worker tasks."""
        for task in self.worker_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                print("Worker task cancelled.")


class AsyncBatchEngine:

    def __init__(
            self,
            processing_function: Callable[[List[Any]], Awaitable[List[Any]]],
            batch_size: int = 32,
            wait_timeout: float = 0.01,
    ):
        """
        Initialize the AsyncBatchEngine with a processing function, number of workers, and batch size.

        Args:
            processing_function (Callable[[List[Any]], Awaitable[List[Any]]]): The batch processing function.
            batch_size (int): The maximum number of requests to process in a single batch.
        """
        self._processing_function = processing_function
        self._batch_size = batch_size
        self._is_running = False
        self._batch_processor = None
        self._wait_timeout = wait_timeout

    async def start(self):
        """Start the engine by initializing the batch processor and worker tasks."""
        if self._is_running:
            return

        self._batch_processor = BatchProcessor(
            processing_function=self._processing_function,
            batch_size=self._batch_size,
            wait_timeout=self._wait_timeout,
            num_workers=1
        )
        self._is_running = True

    async def stop(self):
        """Stop the engine by shutting down the batch processor and worker tasks."""
        self._check_running()
        self._is_running = False
        if self._batch_processor is not None:
            await self._batch_processor.shutdown()

    def _check_running(self):
        """Check if the engine is running.

        Raises:
            ValueError: If the engine is not running.
        """
        if not self._is_running:
            raise ValueError(
                "The engine is not running. "
                "You must start the engine before using it."
            )

    async def add_request(self, single_input: Any, request_id: str = None) -> dict:
        """Asynchronously add a request to be processed.

        Args:
            single_input (Any): The input data for processing.
            request_id (str): Optional request identifier to avoid data mix-up.

        Raises:
            ValueError: If the engine is not running when this method is called.
        """
        if not self._is_running:
            await self.start()

        if request_id is None:
            request_id = str(uuid.uuid4())  # Assign a unique ID if not provided
        future = await self._batch_processor.add_request(single_input=single_input)  # type: ignore
        result = await future
        return dict(
            request_id=request_id,
            feature=result
        )