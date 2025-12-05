# tipc_ext.pyi
# Stub file for the compiled PyTorch extension built from the given pybind11 bindings.

from __future__ import annotations

from typing import List

import torch


class NvIpcWriter:
    """
    Writer for CUDA IPC device memory.

    NvIpcWriter allocates a device buffer on a given CUDA device and
    supports appending (writing) CUDA tensors into this buffer. Once
    all data has been written, build() can be used to export a
    cudaIpcMemHandle_t as a bytes object that can be passed to another
    process and consumed by NvIpcReader.
    """

    def __init__(self, device_id: int, buffer_size: int) -> None:
        """
        Construct a NvIpcWriter and allocate a device buffer.

        Args:
            device_id (int): CUDA device ordinal on which to allocate the buffer.
            buffer_size (int): Size of the internal device buffer in bytes.

        The constructor sets the current CUDA device and allocates a device buffer
        of the given size. The buffer is then used as a contiguous region into
        which tensors are written.
        """
        ...

    def write(self, tensor: torch.Tensor) -> int:
        """
        Write a CUDA tensor's raw bytes into the internal device buffer at the
        current write offset, then advance the offset.

        Args:
            tensor (torch.Tensor): Source CUDA tensor. It must reside on the same
                CUDA device as the writer and be contiguous.

        Returns:
            int: Number of bytes written, or 0 if there is not enough remaining space.

        The copy is enqueued on PyTorch's current CUDA stream for the writer's
        device using cudaMemcpyAsync with cudaMemcpyDeviceToDevice.
        """
        ...

    def build(self) -> bytes:
        """
        Export the device buffer as a CUDA IPC handle string.

        Returns:
            bytes: A string obj containing a serialized cudaIpcMemHandle_t
            corresponding to the writer's device buffer.

        Before exporting, this method synchronizes PyTorch's current CUDA stream
        on the writer's device to ensure all pending async copies into the buffer
        have completed.
        """
        ...

    def reset(self) -> None:
        """
        Reset the internal write offset to the beginning of the buffer.

        After reset(), the next write() call will start writing at offset 0.
        This does not clear or reallocate the underlying device memory.
        """
        ...

    def close(self) -> None:
        """
        Explicitly free the device buffer and reset state.

        After close() is called, the internal buffer pointer is freed and set to
        nullptr. Further write() calls are invalid and should be avoided or guarded
        by the caller.
        """
        ...


class NvIpcReader:
    """
    Reader for CUDA IPC device memory.

    NvIpcReader owns a CUDA IPC handle and an opened device pointer
    corresponding to the exported device memory in another process.
    It implements a read() method to create 1D uint8 CUDA tensors
    that view slices of the shared device buffer.
    """

    def __init__(self, handle: str) -> None:
        """
        Construct a NvIpcReader from an IPC handle string.

        Args:
            handle (str): A string produced by NvIpcWriter.build() in the
                writer process, containing a serialized cudaIpcMemHandle_t.

        The actual opening of the IPC handle (cudaIpcOpenMemHandle) is deferred
        to read(), because the target device_id is specified there.
        """
        ...

    def read(
        self,
        total_bytes: int,
        offsets: List[int],
        device_id: int,
    ) -> List[torch.Tensor]:
        """
        Read slices from the IPC buffer as 1D uint8 CUDA tensors.

        Args:
            total_bytes (int): Total valid bytes in the shared buffer region.
            offsets (List[int]): Starting offsets (in bytes) of each slice.
                Slice i is [offsets[i], offsets[i+1]) for i < len(offsets) - 1,
                and [offsets[-1], total_bytes) for the last slice.
            device_id (int): CUDA device on which to open and use the IPC buffer.

        Returns:
            List[torch.Tensor]: A list of 1D tensors of dtype torch.uint8, each
            tensor representing a slice of the shared buffer.

        Notes:
            - The returned tensors are non-owning views into the underlying IPC
              buffer; their deleter does not free or close the IPC handle.
            - The caller must ensure that NvIpcReader.close() is called only
              after all returned tensors are no longer needed.
        """
        ...

    def close(self) -> None:
        """
        Explicitly close the opened IPC memory handle, if any.

        If read() has opened the IPC handle, this method calls
        cudaIpcCloseMemHandle on the internal device pointer and marks the
        reader as closed. Subsequent calls are no-ops.
        """
        ...


class NvShmReader:
    """
    Reader for shared-memory (shm) files into CUDA tensors.

    NvShmReader maps a shm-backed file into the process address space
    and provides an asynchronous host-to-device copy interface to read
    slices of the shm region into CUDA tensors.
    """

    def __init__(self, shm_file_name: str) -> None:
        """
        Construct a NvShmReader from a shm-backed file.

        Args:
            shm_file_name (str): Path to the shm file, e.g. '/dev/shm/xxx' or
                a file created via shm_open.

        The constructor opens the file, queries its size, and mmaps it into the
        process address space with read-only protection.
        """
        ...

    def read(
        self,
        total_bytes: int,
        offsets: List[int],
        device_id: int,
    ) -> List[torch.Tensor]:
        """
        Read slices from the shm buffer into 1D uint8 CUDA tensors.

        Args:
            total_bytes (int): Total valid bytes in the shm region. Must be > 0 and
                <= the mapped region size.
            offsets (List[int]): Starting offsets (in bytes) of each slice. Slice i is
                [offsets[i], offsets[i+1]) for i < len(offsets)-1, and
                [offsets[-1], total_bytes) for the last slice.
            device_id (int): CUDA device on which the returned tensors should reside.

        Returns:
            List[torch.Tensor]: A list of 1D tensors of dtype torch.uint8, each
            storing a copy of the corresponding slice from shm on the given device.

        Note:
            This implementation performs asynchronous HostToDevice copies and
            returns owning tensors.
        """
        ...

    def close(self) -> None:
        """
        Close the shm mapping and file descriptor.

        After close() is called, further read() calls are invalid.
        """
        ...


class NvShmWriter:
    """
    Writer for shared-memory (shm) files from CUDA tensors.

    NvShmWriter maps a shm-backed file into the process address space
    with write permission and provides an asynchronous device-to-host
    copy interface to write CUDA tensors into the shm region.

    It maintains an internal write offset that is advanced on each write().
    """

    def __init__(self, shm_file_name: str) -> None:
        """
        Construct a NvShmWriter from a shm-backed file.

        Args:
            shm_file_name (str): Path to the shm file, e.g. '/dev/shm/xxx' or
                a file created via shm_open.

        The constructor opens the file, queries its size, and mmaps it into the
        process address space with read/write protection. It also initializes
        the internal write offset to 0.
        """
        ...

    def write(self, tensor: torch.Tensor) -> int:
        """
        Asynchronously copy data from a CUDA tensor to the shm region at the
        current write offset, then advance the offset.

        Args:
            tensor (torch.Tensor): Source CUDA tensor (must be contiguous).

        Returns:
            int: Number of bytes written, or 0 if there is not enough space.

        The copy is enqueued on PyTorch's current CUDA stream associated with
        the tensor's device using cudaMemcpyAsync with cudaMemcpyDeviceToHost.
        Synchronization is the caller's responsibility.
        """
        ...

    def reset(self) -> None:
        """
        Reset the internal write offset to the beginning of the shm region.

        After reset(), the next write() call starts at offset 0. The underlying
        memory is not cleared.
        """
        ...

    def close(self) -> None:
        """
        Close the shm mapping and file descriptor.

        After close(), further write() calls are invalid.
        """
        ...
