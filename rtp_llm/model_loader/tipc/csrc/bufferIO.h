#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

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
     * @param device_id   CUDA device on which the returned tensors should reside.
     *
     * @return std::vector<torch::Tensor>
     *         A list of 1D tensors of dtype torch::kUInt8. The semantics
     *         (view vs. clone, ownership, etc.) are determined by the concrete
     *         implementation.
     */
    virtual std::vector<torch::Tensor>
    read(std::size_t total_bytes, const std::vector<std::int64_t>& offsets, std::int32_t device_id) = 0;

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

}  // namespace tipc