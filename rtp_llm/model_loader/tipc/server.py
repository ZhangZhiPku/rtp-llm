import logging
from typing import List, Optional

import torch

from .core import MethodType, TensorIPCMeta
from .ffi import NvIpcReader, NvShmReader, TipcLib

logger = logging.getLogger(__name__)


class TensorTransportServer:
    """
    Server-side helper to read tensors from shared memory or CUDA IPC.

    It acts as a thin wrapper around `NvShmReader` / `NvIpcReader`, and
    reconstructs tensors based on a list of `TensorIPCMeta` metadata entries.

    Typical usage:
        server = TensorTransportServer(method="shm", storage="/dev/shm/xxx", device_id=0)
        metas = [...]  # list[TensorIPCMeta] previously sent from client
        tensors = server.read(metas)
        server.close()
    """

    def __init__(self, method: MethodType, storage: str):
        """
        Initialize a TensorTransportServer.

        Args:
            method (MethodType):
                Transport method:
                  - "shm": use shared memory backend via NvShmReader
                  - "cuipc": use CUDA IPC backend via NvIpcReader
            storage (str):
                For "shm": shared memory name or path, e.g. "/dev/shm/xxx".
                For "cuipc": CUDA IPC handle identifier or equivalent.

        Raises:
            ValueError: If `method` is not one of {"shm", "cuipc"}.
            RuntimeError: If reader creation fails.
        """
        if method not in {"shm", "cuipc"}:
            raise ValueError(
                f"Unsupported method '{method}', expected 'shm' or 'cuipc'."
            )

        self.storage: str = storage
        self.method: MethodType = method
        self.reader: Optional[NvIpcReader | NvShmReader] = None

        logger.info(
            "Initializing TensorTransportServer: method=%s, storage=%s", method, storage
        )

        try:
            if method == "shm":
                self.reader = TipcLib.NvShmReader(storage)
                logger.debug("NvShmReader created with storage=%s", storage)
            else:  # "cuipc"
                self.reader = TipcLib.NvIpcReader(bytes.fromhex(storage))
                logger.debug("NvIpcReader created with storage=%s", storage)

        except Exception as e:
            logger.exception(
                "Failed to create %s reader for storage=%s",
                "NvShmReader" if method == "shm" else "NvIpcReader",
                storage,
            )
            raise RuntimeError(
                f"Failed to create reader for storage='{storage}' with method='{method}': {e}"
            ) from e

    def read(self, metas: List[TensorIPCMeta], device_id: int) -> List[torch.Tensor]:
        """
        Read a list of tensors from the transport backend using tensor metadata.

        The method:
        - Computes the total number of bytes to read (sum of `meta.size`),
        - Builds an offset array for the reader backend,
        - Calls the underlying reader's `read` and returns the resulting tensors.

        Args:
            metas (list[TensorIPCMeta]):
                Metadata for each tensor to reconstruct. Each `TensorIPCMeta` is
                expected to at least have a `size` attribute, representing the
                number of bytes occupied by the corresponding tensor.

        Returns:
            list[torch.Tensor]: The list of tensors reconstructed from the buffer.

        Raises:
            ValueError: If `metas` list is empty.
            RuntimeError: If the reader has not been initialized or reading fails.
        """
        if not metas:
            raise ValueError("Cannot read tensors: empty metadata list (metas).")

        if self.reader is None:
            raise RuntimeError(
                "Reader is not initialized. Did you encounter an error during construction?"
            )

        # Compute cumulative offsets and total size.
        #
        # Example:
        #   sizes  = [10, 20, 30]
        #   offsets = [0, 10, 30]
        #   total_size = 10 + 20 + 30 = 60
        #
        # Offsets are computed for all tensors except the last, as the last one
        # is implied by the total buffer size.
        offsets: List[int] = [0]
        total_size: int = 0

        for meta in metas[:-1]:
            if meta.size <= 0:
                raise ValueError(f"Invalid tensor size in metadata: {meta.size}")
            total_size += meta.size
            offsets.append(total_size)

        # Add the last tensor size to the total size.
        last_meta = metas[-1]
        if last_meta.size <= 0:
            raise ValueError(f"Invalid tensor size in metadata: {last_meta.size}")
        total_size += last_meta.size

        try:
            _tensors: List[torch.Tensor] = self.reader.read(
                total_size,
                offsets,
                device_id=device_id,
            )
            if len(_tensors) != len(metas):
                raise ValueError("num of tensor mismatchs the num of meta.")

            tensors: List[torch.Tensor] = []
            for meta, tensor in zip(metas, _tensors):
                tensors.append(tensor.view(dtype=meta.dtype).view(size=meta.shape))

        except Exception as e:
            logger.exception(
                "Failed to read tensors from backend (method=%s, storage=%s).",
                self.method,
                self.storage,
            )
            raise RuntimeError(
                f"Failed to read tensors from storage='{self.storage}' "
                f"with method='{self.method}': {e}"
            ) from e

        logger.info(
            "Successfully read %d tensors from backend (method=%s, storage=%s).",
            len(tensors),
            self.method,
            self.storage,
        )

        return tensors

    def close(self) -> None:
        """
        Close the underlying reader and release associated resources.

        This method is idempotent; calling it multiple times is safe.

        Raises:
            RuntimeError: If closing fails.
        """
        if self.reader is None:
            logger.debug("close() called but reader is already None; nothing to do.")
            return

        logger.info(
            "Closing TensorTransportServer reader (method=%s, storage=%s).",
            self.method,
            self.storage,
        )

        try:
            self.reader.close()
        except Exception as e:
            logger.exception(
                "Failed to close reader (method=%s, storage=%s).",
                self.method,
                self.storage,
            )
            raise RuntimeError(
                f"Failed to close reader for storage='{self.storage}' "
                f"with method='{self.method}': {e}"
            ) from e
        finally:
            self.reader = None

    def __del__(self):
        """
        Destructor: best-effort attempt to close the reader.

        Exceptions are intentionally suppressed since raising exceptions from
        `__del__` is discouraged and can cause issues during interpreter shutdown.
        """
        try:
            self.close()
        except Exception:
            # Suppress all errors in destructor
            pass
