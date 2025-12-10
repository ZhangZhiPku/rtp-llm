#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <torch/extension.h>

namespace tipc {

/**
 * @brief Abstract interface for reading tensors from a buffered backend.
 *
 * The typical use case is reading tensors from a contiguous buffer (e.g.
 * CUDA IPC buffer, shared memory buffer, etc.) into CUDA tensors.
 */
class BufferedTensorReader {
public:
    virtual ~BufferedTensorReader() = default;

    /**
     * @brief Read 1D uint8 tensors from an underlying buffer.
     *
     * @param total_bytes Total valid bytes in the buffer region that can be read.
     * @param offsets     Starting offsets (in bytes) of each slice to read.
     *                    The length of each slice i is:
     *                      - offsets[i+1] - offsets[i] for i < offsets.size() - 1
     *                      - total_bytes - offsets.back() for the last slice.
     * @param device_id   CUDA device pcie address.
     *
     * @return std::vector<torch::Tensor>
     *         A list of 1D tensors of dtype torch::kUInt8. The semantics
     *         (view vs. clone, ownership, etc.) are determined by the concrete
     *         implementation.
     */
    virtual std::vector<torch::Tensor>
    read(std::size_t total_bytes, const std::vector<std::int64_t>& offsets, std::string device_pcie_str) = 0;

    /**
     * @brief Close the underlying resource (file descriptor, IPC handle, etc.).
     *
     * Implementations should release any OS- or CUDA-level handles they own.
     * After close() has been called, further read() calls are typically invalid
     * or should throw.
     */
    virtual void close() = 0;
};

/**
 * @brief Abstract interface for writing tensors into a buffered backend.
 *
 * The typical use case is writing tensors into a contiguous buffer (e.g.
 * CUDA IPC buffer, shared memory buffer, etc.) from CUDA tensors.
 *
 * The implementation is expected to maintain an internal write offset.
 * Each call to write() writes at the current offset and advances it by
 * the number of bytes written.
 */
class BufferedTensorWriter {
public:
    virtual ~BufferedTensorWriter() = default;

    /**
     * @brief Write a tensor into the underlying buffer at the current write offset.
     *
     * Implementations should:
     *  - write the raw bytes of @p t starting at the internal offset,
     *  - advance the internal offset by t.nbytes(),
     *  - return the number of bytes actually written.
     *
     * @param t  Source tensor to write. Usually a CUDA tensor.
     *
     * @return std::size_t
     *         Number of bytes written. Implementations may return 0 or a
     *         sentinel value (e.g. std::size_t(-1)) to indicate failure.
     */
    virtual std::size_t write(const torch::Tensor& t) = 0;

    /**
     * @brief Reset the internal write offset to the beginning of the buffer.
     *
     * After reset(), the next write() call should start writing at offset 0
     * (or the implementation-defined initial position).
     */
    virtual void reset() = 0;

    /**
     * @brief Close the underlying resource (file descriptor, device buffer, etc.).
     *
     * Implementations should release any OS- or CUDA-level handles they own.
     * After close() has been called, further write() calls are typically invalid
     * or should throw.
     */
    virtual void close() = 0;
};

/**
 * @brief Get the PCIe bus identifier string of the CUDA device on which
 *        a given tensor resides.
 *
 * This function inspects the device of the input tensor @p t, obtains its
 * logical CUDA device index (e.g. 0, 1, ...), and then queries CUDA for the
 * PCIe bus ID string corresponding to that device (typically in a form like
 * "0000:81:00.0").
 *
 * The returned PCIe string is a *physical* identifier of the GPU in the
 * system and is independent of any CUDA device indexing remapping that may
 * be applied via @c CUDA_VISIBLE_DEVICES.
 *
 * @note
 * In CUDA IPC or multi-process scenarios, this PCIe string is useful for
 * correctly matching GPUs across different processes:
 *
 * - Each process may have a different @c CUDA_VISIBLE_DEVICES setting.
 *   As a result, "device 0" in process A may refer to a completely
 *   different physical GPU than "device 0" in process B.
 *
 * - If we only transmit a logical device index (e.g. "0") derived from
 *   a tensor's device, the receiving process might interpret that index
 *   according to its own @c CUDA_VISIBLE_DEVICES mapping and end up using
 *   the wrong GPU.
 *
 * - To avoid this ambiguity, we transmit the PCIe bus ID string of the
 *   GPU that actually holds the tensor's data. The receiver can then:
 *      1. Enumerate its own visible CUDA devices.
 *      2. For each device, query its PCIe bus ID.
 *      3. Find the local device whose PCIe bus ID matches the one provided
 *         by the sender.
 *      4. Set that device as the active CUDA device and then open/use any
 *         CUDA IPC memory handles or perform peer access on the correct
 *         physical GPU.
 *
 * In short, the PCIe string acts as a stable, process-independent
 * identifier that allows different processes, possibly with different
 * CUDA-visible device orders, to agree on and select the same physical
 * GPU when sharing memory or tensor data.
 *
 * @param t
 *      A tensor that must reside on a CUDA device (i.e. t.is_cuda() == true).
 *
 * @return std::string
 *      PCIe bus ID string for the GPU on which @p t resides
 *      (e.g. "0000:81:00.0").
 *
 * @throws std::runtime_error
 *      If @p t is not a CUDA tensor, or querying its PCIe bus ID fails.
 */
std::string get_tensor_device_pcie_str(const torch::Tensor& t) {
    if (!t.is_cuda()) {
        throw std::runtime_error("get_tensor_device_pcie_str: tensor is not on CUDA device");
    }

    // Logical CUDA device index for this tensor
    int device_id = t.get_device();

    char        bus_id[13] = {0};
    cudaError_t st         = cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), device_id);
    if (st != cudaSuccess) {
        throw std::runtime_error(std::string("get_tensor_device_pcie_str: cudaDeviceGetPCIBusId failed: ")
                                 + cudaGetErrorString(st));
    }

    st = cudaDeviceGetByPCIBusId(&device_id, bus_id);

    if (st != cudaSuccess) {
        throw std::runtime_error(std::string("not cool: ") + cudaGetErrorString(st));
    }

    return std::string(bus_id);
}

}  // namespace tipc