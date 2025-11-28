import asyncio
import threading
import time
import traceback
from dataclasses import dataclass
from logging import getLogger
from typing import List, Optional

import grpc.aio
import torch

from .model_rpc_service_pb2 import GenerateInputPB, GenerateOutputsPB, TensorPB
from .model_rpc_service_pb2_grpc import RpcServiceStub

logger = getLogger()


@dataclass
class LLMRequest:
    """Data class to hold Large Language Model (LLM) request parameters."""

    tokens: list[int]
    rmp_tokens: list[int]  # tokens after remove padding
    attention_mask: list
    position_ids: list
    request_id: int = 0
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    max_new_tokens: int = 1000
    n: int = 1
    timestamp: float = 0.0  # Timestamp when the request was sent
    roll_request_id: str = ""
    stop_token_ids: Optional[List[int]] = None


@dataclass
class LLMResponse:
    """Data class to hold Large Language Model (LLM) response parameters."""

    tokens: torch.Tensor
    logits: torch.Tensor
    timestamp: float
    error: Optional[str] = None
    status_code: int = 0  # For gRPC, this might represent a gRPC status code, 0 for OK


@dataclass
class RequestLog:
    """Data class to log each request and its corresponding response."""

    request: LLMRequest
    response: LLMResponse


def _convert_tensor_pb_to_torch(tensor_pb: TensorPB) -> torch.Tensor:
    """
    Converts a TensorPB (protobuf) object to a torch.Tensor.
    Args:
        tensor_pb (TensorPB): The protobuf tensor object.
    Returns:
        torch.Tensor: The converted PyTorch tensor.
    Raises:
        Exception: If the data type is unknown.
    """
    if not (len(tensor_pb.shape) > 0 and tensor_pb.shape[0] > 0):
        return torch.tensor([], dtype=torch.float32)
    if tensor_pb.data_type == TensorPB.DataType.FP32:
        return torch.frombuffer(tensor_pb.fp32_data, dtype=torch.float32).reshape(
            list(tensor_pb.shape)
        )
    elif tensor_pb.data_type == TensorPB.DataType.INT32:
        return torch.frombuffer(tensor_pb.int32_data, dtype=torch.int32).reshape(
            list(tensor_pb.shape)
        )
    elif tensor_pb.data_type == TensorPB.DataType.FP16:
        return torch.frombuffer(tensor_pb.fp16_data, dtype=torch.float16).reshape(
            list(tensor_pb.shape)
        )
    elif tensor_pb.data_type == TensorPB.DataType.BF16:
        return torch.frombuffer(tensor_pb.bf16_data, dtype=torch.bfloat16).reshape(
            list(tensor_pb.shape)
        )
    else:
        raise Exception("Unknown tensor data type")


def build_output_py(output_pb: GenerateOutputsPB) -> LLMResponse:
    """
    Builds an LLMResponse object from a GenerateOutputsPB (protobuf) object.
    Args:
        output_pb (GenerateOutputsPB): The gRPC response protobuf for a single output.
    Returns:
        LLMResponse: The constructed LLM response object.
    """
    return LLMResponse(
        tokens=_convert_tensor_pb_to_torch(output_pb.output_ids).flatten(),
        logits=_convert_tensor_pb_to_torch(output_pb.logits),
        timestamp=time.time(),
        error=None,  # No error for successful output
        status_code=0,  # OK status
    )


def build_input_pb(request: LLMRequest) -> GenerateInputPB:
    """
    Builds the input protobuf for the gRPC request.
    Args:
        request (LLMRequest): The LLM request object.
    Returns:
        GenerateInputPB: The constructed input protobuf.
    """
    input_pb = GenerateInputPB()
    input_pb.request_id = request.request_id
    input_pb.token_ids.extend(request.rmp_tokens)
    generate_config_pb = input_pb.generate_config
    generate_config_pb.num_return_sequences = request.n
    generate_config_pb.max_new_tokens = request.max_new_tokens
    generate_config_pb.top_k = request.top_k
    generate_config_pb.top_p = request.top_p
    generate_config_pb.temperature = request.temperature
    generate_config_pb.repetition_penalty = 1.0
    generate_config_pb.num_beams = 1
    generate_config_pb.do_sample = True
    for token_id in request.stop_token_ids:
        stop_word = generate_config_pb.stop_words_list.rows.add()
        stop_word.values.append(token_id)
    return input_pb


class AsyncGrpcClient:
    """gRPC client for asynchronous operations."""

    def __init__(self, target_url: str, concurrency_limit: int = 1024):
        self.target_url = target_url
        self.running: bool = True
        self.request_logs: list[RequestLog] = []
        self.lock = threading.Lock()  # Protects access to request_logs
        self.concurrency_limit = concurrency_limit
        self._worker_loop_event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_send_queue: Optional[asyncio.Queue] = (
            None  # Will be initialized in the worker thread
        )
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[RpcServiceStub] = None
        # The worker thread needs to run an asyncio event loop.
        self.worker = threading.Thread(target=self._run_worker_loop, daemon=True)
        self.worker.start()
        self.GRACEFUL_SHUTDOWN_TIMEOUT = 60  # seconds
        # Wait briefly for the worker thread to initialize its event loop and queue
        time.sleep(0.1)
        if self._worker_loop_event_loop is None or self._async_send_queue is None:
            raise RuntimeError("Failed to initialize worker event loop or queue.")

    def _run_worker_loop(self):
        """Helper method to run the asyncio event loop in the worker thread."""
        self._worker_loop_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._worker_loop_event_loop)
        # Initialize asyncio.Queue within the event loop it will be used
        self._async_send_queue = asyncio.Queue()
        # Initialize gRPC channel and stub
        # Using grpc.aio.insecure_channel for demonstration. Use secure_channel for production.
        self._channel = grpc.aio.insecure_channel(self.target_url)
        self._stub = RpcServiceStub(self._channel)
        self._worker_loop_event_loop.run_until_complete(self._worker_loop())
        self._worker_loop_event_loop.close()

    async def _send_grpc_request(self, request: LLMRequest) -> list[RequestLog]:
        """
        Sends a gRPC request to the server and returns multiple RequestLog.
        Handles gRPC specific errors and processes server-streaming RPC responses.
        We use `num_return_sequences` to control how many results a single request should return,
        which is useful in the reinforcement learning rollout phase for controlling sample quantity.
        """
        results: list[RequestLog] = []
        try:
            input_pb = build_input_pb(request)
            # Iterate over the streaming responses from the server
            async for grpc_response_pb in self._stub.GenerateStreamCall(input_pb):
                # There are 'n' outputs from one input sample.
                for output in grpc_response_pb.generate_outputs:
                    if not output.finished:
                        continue  # Skip unfinished requests.
                    results.append(
                        RequestLog(request=request, response=build_output_py(output))
                    )
        except grpc.aio.AioRpcError as e:
            error_message = f"gRPC Error (code: {e.code().name}): {e.details()}\n{traceback.format_exc()}"
            status_code = e.code().value
            results = [
                RequestLog(
                    request,
                    LLMResponse(
                        tokens=torch.tensor([], dtype=torch.int32),
                        logits=torch.tensor([], dtype=torch.float32),
                        timestamp=time.time(),
                        error=error_message,
                        status_code=status_code,
                    ),
                )
            ]
        except Exception as e:
            error_message = (
                f"An unexpected error occurred: {e}\n{traceback.format_exc()}"
            )
            results = [
                RequestLog(
                    request,
                    LLMResponse(
                        tokens=torch.tensor([], dtype=torch.int32),
                        logits=torch.tensor([], dtype=torch.float32),
                        timestamp=time.time(),
                        error=error_message,
                        status_code=grpc.StatusCode.UNKNOWN.value,  # Use UNKNOWN for general exceptions
                    ),
                )
            ]
        return results

    def enqueue(self, request: LLMRequest) -> None:
        """
        This function is called by the main thread to add a request to the execution queue.
        It schedules a coroutine on the worker's asyncio event loop to process the request.
        """
        if (
            not self._worker_loop_event_loop
            or not self._worker_loop_event_loop.is_running()
        ):
            raise RuntimeError(
                "Worker event loop is not running. Cannot enqueue request."
            )
        request.timestamp = time.time()  # Mark the time when the request is enqueued
        try:
            asyncio.run_coroutine_threadsafe(
                self._async_send_queue.put(request), self._worker_loop_event_loop
            )
        except asyncio.CancelledError:
            raise RuntimeError(
                "Failed to enqueue request: Worker loop is shutting down."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to enqueue request: {e}")

    async def _worker_loop(self):
        """
        The worker loop is the main loop of the worker thread.
        It continuously retrieves requests from the asyncio queue and sends them to the server.
        """
        # Create a Semaphore to limit the number of concurrent requests
        semaphore = asyncio.Semaphore(self.concurrency_limit)

        async def process_single_request(request: LLMRequest):
            async with semaphore:
                records = await self._send_grpc_request(request)
                with self.lock:
                    self.request_logs.extend(records)
                self._async_send_queue.task_done()

        while self.running:
            requests_to_process: List[LLMRequest] = []
            try:
                # Attempt to get up to concurrency_limit requests from the queue without blocking
                # This helps in batching the execution with asyncio.gather
                for _ in range(self.concurrency_limit):
                    try:
                        # Use a small timeout to avoid indefinitely waiting if no requests arrive
                        request = await asyncio.wait_for(
                            self._async_send_queue.get(), timeout=0.1
                        )
                        requests_to_process.append(request)
                    except asyncio.TimeoutError:
                        break  # No more requests currently in queue
                if not requests_to_process:
                    # If no requests were collected, yield control to the event loop
                    # This prevents a busy loop if the queue is empty
                    await asyncio.sleep(0.01)
                    continue
                # Concurrently process all collected requests
                await asyncio.gather(
                    *[process_single_request(req) for req in requests_to_process]
                )
            except Exception as e:
                # Catch any unexpected errors that might occur during the batch processing
                logger.info(f"Error in worker loop: {e}")
                # For any requests that were pulled but failed to process due to a general loop error,
                # you might want to log them as errors.
                for req in requests_to_process:
                    with self.lock:
                        self.request_logs.append(
                            RequestLog(
                                request=req,
                                response=LLMResponse(
                                    tokens=torch.tensor([], dtype=torch.int32),
                                    logits=torch.tensor([], dtype=torch.float32),
                                    timestamp=time.time(),
                                    error=f"Worker processing error: {e}",
                                    status_code=grpc.StatusCode.UNKNOWN.value,
                                ),
                            )
                        )
                    self._async_send_queue.task_done()  # Mark task as done even on error

    def collect(self) -> list[RequestLog]:
        """
        The `collect` function is used to gather all request records processed by the worker thread.
        It returns a list containing all `RequestLog` objects and then clears the current record list.
        """
        with self.lock:
            logs = self.request_logs.copy()
            self.request_logs.clear()
        return logs

    def close(self):
        """Closes the worker thread and cleans up resources."""
        logger.info("Closing AsyncGrpcClient...")
        self.running = False  # Signal the worker loop to stop
        if self._worker_loop_event_loop and self._worker_loop_event_loop.is_running():
            # Create a future and run it on the worker's event loop to close the channel.
            # This ensures the channel.close() is awaited in the correct thread.
            future = asyncio.run_coroutine_threadsafe(
                self._channel.close(), self._worker_loop_event_loop
            )
            try:
                # Wait for the channel to close, with a timeout
                future.result(
                    timeout=self.GRACEFUL_SHUTDOWN_TIMEOUT / 2
                )  # Give it half the timeout
                logger.info("gRPC channel closed successfully.")
            except asyncio.TimeoutError:
                logger.info("Warning: Timed out waiting for gRPC channel to close.")
            except Exception as e:
                logger.info(f"Error closing gRPC channel: {e}")
        # Wait for the worker thread to finish its loop after self.running is False
        if self.worker.is_alive():
            logger.info(
                f"Waiting for worker thread to finish (timeout: {self.GRACEFUL_SHUTDOWN_TIMEOUT}s)..."
            )
            self.worker.join(timeout=self.GRACEFUL_SHUTDOWN_TIMEOUT)
        if self.worker.is_alive():
            logger.info("Warning: Worker thread did not terminate gracefully.")
        else:
            logger.info("Worker thread terminated.")
