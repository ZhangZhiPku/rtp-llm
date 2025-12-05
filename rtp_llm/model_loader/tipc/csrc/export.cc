#include "common.h"
#include "shm.h"
#include "ipc.h"

using namespace tipc;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // -------- NvIpcWriter --------
    py::class_<NvIpcWriter>(m, "NvIpcWriter")
        .def(py::init<std::int32_t, std::size_t>(),
             py::arg("device_id"),
             py::arg("buffer_size"),
             "Construct a NvIpcWriter and allocate a device buffer.\n\n"
             "Args:\n"
             "    device_id (int): CUDA device ordinal on which to allocate the buffer.\n"
             "    buffer_size (int): Size of the internal device buffer in bytes.\n\n"
             "The constructor sets the current CUDA device and allocates a device buffer\n"
             "of the given size. The buffer is then used as a contiguous region into\n"
             "which tensors are written.")
        .def("write",
             &NvIpcWriter::write,
             py::arg("tensor"),
             "Write a CUDA tensor's raw bytes into the internal device buffer at the\n"
             "current write offset, then advance the offset.\n\n"
             "Args:\n"
             "    tensor (torch.Tensor): Source CUDA tensor. It must reside on the same\n"
             "        CUDA device as the writer and be contiguous.\n\n"
             "Returns:\n"
             "    int: Number of bytes written, or 0 if there is not enough remaining space.\n\n"
             "The copy is enqueued on PyTorch's current CUDA stream for the writer's\n"
             "device using cudaMemcpyAsync with cudaMemcpyDeviceToDevice.")
        .def("build",
             &NvIpcWriter::build,
             "Export the device buffer as a CUDA IPC handle string.\n\n"
             "Returns:\n"
             "    bytes: A binary blob containing a serialized cudaIpcMemHandle_t\n"
             "           corresponding to the writer's device buffer.\n\n"
             "Before exporting, this method synchronizes PyTorch's current CUDA stream\n"
             "on the writer's device to ensure all pending async copies into the buffer\n"
             "have completed.")
        .def("reset",
             &NvIpcWriter::reset,
             "Reset the internal write offset to the beginning of the buffer.\n\n"
             "After reset(), the next write() call will start writing at offset 0.\n"
             "This does not clear or reallocate the underlying device memory.")
        .def("close",
             &NvIpcWriter::close,
             "Explicitly free the device buffer and reset state.\n\n"
             "After close() is called, the internal buffer pointer is freed and set to\n"
             "nullptr. Further write() calls are invalid and should be avoided or guarded\n"
             "by the caller.");

    // -------- NvIpcReader --------
    py::class_<NvIpcReader>(m, "NvIpcReader")
        .def(py::init<const std::string&>(),
             py::arg("handle"),
             "Construct a NvIpcReader from an IPC handle string.\n\n"
             "Args:\n"
             "    handle (bytes): A binary blob produced by NvIpcWriter.build() in the\n"
             "        writer process, containing a serialized cudaIpcMemHandle_t.\n\n"
             "The actual opening of the IPC handle (cudaIpcOpenMemHandle) is deferred\n"
             "to read(), because the target device_id is specified there.")
        .def("read",
             &NvIpcReader::read,
             py::arg("total_bytes"),
             py::arg("offsets"),
             py::arg("device_id"),
             "Read slices from the IPC buffer as 1D uint8 CUDA tensors.\n\n"
             "Args:\n"
             "    total_bytes (int): Total valid bytes in the shared buffer region.\n"
             "    offsets (List[int]): Starting offsets (in bytes) of each slice.\n"
             "        Slice i is [offsets[i], offsets[i+1]) for i < len(offsets) - 1,\n"
             "        and [offsets[-1], total_bytes) for the last slice.\n"
             "    device_id (int): CUDA device on which to open and use the IPC buffer.\n\n"
             "Returns:\n"
             "    List[torch.Tensor]: A list of 1D tensors of dtype torch.uint8, each\n"
             "    tensor representing a slice of the shared buffer.\n\n"
             "Notes:\n"
             "    - The returned tensors are non-owning views into the underlying IPC\n"
             "      buffer; their deleter does not free or close the IPC handle.\n"
             "    - The caller must ensure that NvIpcReader.close() is called only\n"
             "      after all returned tensors are no longer needed.")
        .def("close",
             &NvIpcReader::close,
             "Explicitly close the opened IPC memory handle, if any.\n\n"
             "If read() has opened the IPC handle, this method calls\n"
             "cudaIpcCloseMemHandle on the internal device pointer and marks the\n"
             "reader as closed. Subsequent calls are no-ops.");

    // -------- NvShmReader --------
    py::class_<NvShmReader>(m, "NvShmReader")
        .def(py::init<const std::string&>(),
             py::arg("shm_file_name"),
             "Construct a NvShmReader from a shm-backed file.\n\n"
             "Args:\n"
             "    shm_file_name (str): Path to the shm file, e.g. '/dev/shm/xxx' or\n"
             "        a file created via shm_open.\n\n"
             "The constructor opens the file, queries its size, and mmaps it into the\n"
             "process address space with read-only protection.")
        .def("read",
             &NvShmReader::read,
             py::arg("total_bytes"),
             py::arg("offsets"),
             py::arg("device_id"),
             "Read slices from the shm buffer into 1D uint8 CUDA tensors.\n\n"
             "Args:\n"
             "    total_bytes (int): Total valid bytes in the shm region. Must be > 0 and\n"
             "        <= the mapped region size.\n"
             "    offsets (List[int]): Starting offsets (in bytes) of each slice. Slice i is\n"
             "        [offsets[i], offsets[i+1]) for i < len(offsets)-1, and\n"
             "        [offsets[-1], total_bytes) for the last slice.\n"
             "    device_id (int): CUDA device on which the returned tensors should reside.\n\n"
             "Returns:\n"
             "    List[torch.Tensor]: A list of 1D tensors of dtype torch.uint8, each\n"
             "    storing a copy of the corresponding slice from shm on the given device.\n\n"
             "Note:\n"
             "    This implementation performs asynchronous HostToDevice copies and\n"
             "    returns owning tensors.")
        .def("close",
             &NvShmReader::close,
             "Close the shm mapping and file descriptor.\n\n"
             "After close() is called, further read() calls are invalid.");

    // -------- NvShmWriter --------
    py::class_<NvShmWriter>(m, "NvShmWriter")
        .def(py::init<const std::string&>(),
             py::arg("shm_file_name"),
             "Construct a NvShmWriter from a shm-backed file.\n\n"
             "Args:\n"
             "    shm_file_name (str): Path to the shm file, e.g. '/dev/shm/xxx' or\n"
             "        a file created via shm_open.\n\n"
             "The constructor opens the file, queries its size, and mmaps it into the\n"
             "process address space with read/write protection. It also initializes\n"
             "the internal write offset to 0.")
        .def("write",
             &NvShmWriter::write,
             py::arg("tensor"),
             "Asynchronously copy data from a CUDA tensor to the shm region at the\n"
             "current write offset, then advance the offset.\n\n"
             "Args:\n"
             "    tensor (torch.Tensor): Source CUDA tensor (must be contiguous).\n\n"
             "Returns:\n"
             "    int: Number of bytes written, or 0 if there is not enough space.\n\n"
             "The copy is enqueued on PyTorch's current CUDA stream associated with\n"
             "the tensor's device using cudaMemcpyAsync with cudaMemcpyDeviceToHost.\n"
             "Synchronization is the caller's responsibility.")
        .def("reset",
             &NvShmWriter::reset,
             "Reset the internal write offset to the beginning of the shm region.\n\n"
             "After reset(), the next write() call starts at offset 0. The underlying\n"
             "memory is not cleared.")
        .def("close",
             &NvShmWriter::close,
             "Close the shm mapping and file descriptor.\n\n"
             "After close(), further write() calls are invalid.");
}