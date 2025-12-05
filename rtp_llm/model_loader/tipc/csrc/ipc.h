#pragma once

#include "common.h"
#include "bufferIO.h"

#include <string>
#include <stdexcept>
#include <vector>
#include <cstring>

namespace py = pybind11;

namespace tipc {

/**
 * @brief Reader for CUDA IPC device memory.
 *
 * NvIpcReader owns a CUDA IPC handle and an opened device pointer
 * corresponding to the exported device memory in another process.
 * It implements BufferedTensorReader::read() to create 1D uint8 CUDA
 * tensors that view slices of the shared device buffer.
 *
 * Lifecycle:
 *  - Construct from IPC handle (std::string).
 *  - Use read() zero or more times to obtain tensor views.
 *  - Call close() to release the opened IPC memory handle.
 *    The destructor will also call close() if it has not been called.
 */
class NvIpcReader: public BufferedTensorReader {
public:
    /**
     * @brief Construct a new NvIpcReader from an IPC handle string.
     *
     * @param handle   A binary blob produced by NvIpcWriter::build() in
     *                 the writer process, containing a serialized
     *                 cudaIpcMemHandle_t.
     *
     * The actual opening of the IPC handle (cudaIpcOpenMemHandle) is
     * deferred to read(), because the device_id is only known there.
     */
    explicit NvIpcReader(const py::bytes& handle): handle_{}, dev_ptr_(nullptr), opened_(false) {
        std::string buf = handle;
        if (buf.size() != sizeof(cudaIpcMemHandle_t)) {
            throw std::runtime_error("NvIpcReader: invalid IPC handle size");
        }
        std::memcpy(&handle_, buf.data(), sizeof(cudaIpcMemHandle_t));
    }

    /**
     * @brief Destroy the NvIpcReader object.
     *
     * If the IPC memory handle has been opened and not yet closed,
     * this destructor will call cudaIpcCloseMemHandle on the device
     * pointer.
     */
    ~NvIpcReader() override {
        try {
            close();
        } catch (...) {
            // Destructors must not throw; swallow any errors.
        }
    }

    /**
     * @brief Explicitly close the opened IPC memory handle, if any.
     *
     * If read() has opened the IPC handle, this method calls
     * cudaIpcCloseMemHandle on the internal device pointer and
     * marks the reader as closed. Subsequent calls are no-ops.
     */
    void close() override {
        if (opened_ && dev_ptr_ != nullptr) {
            cudaError_t st = cudaIpcCloseMemHandle(dev_ptr_);
            (void)st;
            dev_ptr_ = nullptr;
            opened_  = false;
        }
    }

    /**
     * @brief Implementation of BufferedTensorReader::read().
     *
     * Read slices from the IPC buffer as 1D uint8 CUDA tensors.
     *
     * @param total_bytes  Total valid bytes in the shared buffer region.
     *                     Must be >= offsets.back() and > 0.
     * @param offsets      Starting offsets (in bytes) of each slice.
     * @param device_id    CUDA device on which to open and use the IPC buffer.
     *
     * @return std::vector<torch::Tensor>
     *         A list of 1D tensors of dtype torch::kUInt8, each tensor
     *         representing a slice [offsets[i], offsets[i+1]) (or
     *         [offsets[i], total_bytes) for the last slice).
     *
     * Notes:
     *  - The returned tensors are non-owning views into the underlying IPC
     *    buffer; their deleter does not free or close the IPC handle.
     *  - The caller must ensure that NvIpcReader::close() is called only
     *    after all returned tensors are no longer needed.
     */
    std::vector<torch::Tensor>
    read(std::size_t total_bytes, const std::vector<std::int64_t>& offsets, std::int32_t device_id) override {
        const int64_t occupied_bytes = static_cast<int64_t>(total_bytes);

        if (occupied_bytes <= 0) {
            throw std::runtime_error("NvIpcReader::read: total_bytes must be positive");
        }

        if (offsets.empty()) {
            return {};
        }

        cudaError_t st = cudaSetDevice(device_id);
        if (st != cudaSuccess) {
            throw std::runtime_error(std::string("NvIpcReader::read: cudaSetDevice failed: ") + cudaGetErrorString(st));
        }

        // Open IPC handle on first use.
        if (!opened_) {
            st = cudaIpcOpenMemHandle(&dev_ptr_, handle_, cudaIpcMemLazyEnablePeerAccess);
            if (st != cudaSuccess) {
                throw std::runtime_error(std::string("NvIpcReader::read: cudaIpcOpenMemHandle failed: ")
                                         + cudaGetErrorString(st));
            }
            opened_ = true;
        }

        std::vector<torch::Tensor> result;
        result.reserve(offsets.size());

        auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device_id);

        for (std::size_t i = 0; i < offsets.size(); ++i) {
            int64_t start = offsets[i];
            if (start < 0 || static_cast<std::uint64_t>(start) >= static_cast<std::uint64_t>(occupied_bytes)) {
                throw std::runtime_error("NvIpcReader::read: offset out of range");
            }

            int64_t end = (i + 1 < offsets.size()) ? offsets[i + 1] : occupied_bytes;
            if (end <= start || static_cast<std::uint64_t>(end) > static_cast<std::uint64_t>(occupied_bytes)) {
                throw std::runtime_error("NvIpcReader::read: invalid offset range");
            }

            int64_t              len        = end - start;
            void*                tensor_ptr = static_cast<char*>(dev_ptr_) + start;
            std::vector<int64_t> sizes{len};

            // Empty deleter: we do not close the IPC handle or free memory here.
            auto t = torch::from_blob(tensor_ptr, sizes, [](void* /*p*/) {}, opts);

            result.push_back(t);
        }

        return result;
    }

private:
    cudaIpcMemHandle_t handle_;
    void*              dev_ptr_;
    bool               opened_;
};

/**
 * @brief Writer for CUDA IPC device memory.
 *
 * NvIpcWriter allocates a device buffer on a given CUDA device and
 * supports appending (pushing) CUDA tensors into this buffer. Once
 * all data has been written, build() can be used to export a
 * cudaIpcMemHandle_t as a string that can be passed to another
 * process and consumed by NvIpcReader.
 *
 * The internal write offset is maintained by occupied_size_.
 */
class NvIpcWriter: public BufferedTensorWriter {
public:
    /**
     * @brief Construct a new NvIpcWriter and allocate device memory.
     *
     * @param device_id   CUDA device ordinal on which to allocate the buffer.
     * @param buffer_size Size of the device buffer in bytes.
     *
     * This constructor sets the current device to @p device_id and
     * allocates @p buffer_size bytes of device memory. The buffer is
     * used as a contiguous region into which tensors are written.
     */
    NvIpcWriter(const std::int32_t device_id, const std::size_t buffer_size):
        occupied_size_(0), buffer_size_(buffer_size), buffer_(nullptr), device_id_(device_id) {

        cudaError_t st = cudaSetDevice(device_id_);
        if (st != cudaSuccess) {
            throw std::runtime_error(std::string("NvIpcWriter: cudaSetDevice failed: ") + cudaGetErrorString(st));
        }
        st = cudaMalloc(&buffer_, buffer_size_);
        if (st != cudaSuccess) {
            throw std::runtime_error(std::string("NvIpcWriter: cudaMalloc failed: ") + cudaGetErrorString(st));
        }
    }

    /**
     * @brief Destroy the NvIpcWriter and free the device buffer.
     *
     * If a device buffer was successfully allocated, this destructor
     * will set the current device to the writer's device_id and call
     * cudaFree on the buffer pointer.
     */
    ~NvIpcWriter() override {
        try {
            close();
        } catch (...) {
            // do not throw from destructor
        }
    }

    /**
     * @brief Implementation of BufferedTensorWriter::write().
     *
     * Write a CUDA tensor's contents into the internal device buffer at
     * the current write offset and advance the offset.
     *
     * @param t  Source CUDA tensor. It must:
     *           - reside on the writer's device (device_id_),
     *           - be contiguous.
     *
     * @return std::size_t
     *         Number of bytes written, or 0 if there is not enough space.
     *
     * The copy is enqueued on PyTorch's current CUDA stream for the writer's
     * device using cudaMemcpyAsync with cudaMemcpyDeviceToDevice.
     */
    std::size_t write(const torch::Tensor& t) override {
        TORCH_CHECK(t.is_cuda(), "NvIpcWriter::write: tensor must be on CUDA device");
        TORCH_CHECK(t.device().index() == device_id_,
                    "NvIpcWriter::write: tensor device mismatch with NvIpcWriter device");
        TORCH_CHECK(t.is_contiguous(), "NvIpcWriter::write: tensor must be contiguous");

        cudaError_t st = cudaSetDevice(device_id_);
        if (st != cudaSuccess) {
            throw std::runtime_error(std::string("NvIpcWriter::write: cudaSetDevice failed: ")
                                     + cudaGetErrorString(st));
        }

        std::size_t bytes = static_cast<std::size_t>(t.numel()) * t.element_size();
        if (occupied_size_ + bytes > buffer_size_) {
            return 0;
        }

        void*       dst = static_cast<char*>(buffer_) + occupied_size_;
        const void* src = t.data_ptr();

        at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream(device_id_);
        cudaStream_t         stream       = torch_stream.stream();

        st = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
        if (st != cudaSuccess) {
            throw std::runtime_error(std::string("NvIpcWriter::write: cudaMemcpyAsync failed: ")
                                     + cudaGetErrorString(st));
        }

        occupied_size_ += bytes;
        return bytes;
    }

    /**
     * @brief Reset the internal write offset to the beginning of the buffer.
     *
     * After reset(), the next write() call will start writing at offset 0.
     * This does not clear or reallocate the underlying device memory.
     */
    void reset() override {
        occupied_size_ = 0;
    }

    /**
     * @brief Explicitly free the device buffer and reset state.
     *
     * After close() is called, the internal buffer pointer is freed and
     * set to nullptr. Further write() calls are invalid and should be
     * avoided or guarded by the caller.
     */
    void close() override {
        if (buffer_ != nullptr) {
            cudaSetDevice(device_id_);
            cudaFree(buffer_);
            buffer_        = nullptr;
            occupied_size_ = 0;
            buffer_size_   = 0;
        }
    }

    /**
     * @brief Export the device buffer as a CUDA IPC handle string.
     *
     * @return std::string  A binary blob containing a serialized
     *                      cudaIpcMemHandle_t corresponding to the
     *                      writer's device buffer.
     *
     * Before exporting, this method synchronizes PyTorch's current
     * CUDA stream on the writer's device to ensure all pending async
     * copies into the buffer have completed.
     */
    py::bytes build() {
        cudaError_t st = cudaSetDevice(device_id_);
        if (st != cudaSuccess) {
            throw std::runtime_error(std::string("NvIpcWriter::build: cudaSetDevice failed: ")
                                     + cudaGetErrorString(st));
        }

        at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream(device_id_);
        cudaStream_t         stream       = torch_stream.stream();
        st                                = cudaStreamSynchronize(stream);
        if (st != cudaSuccess) {
            throw std::runtime_error(std::string("NvIpcWriter::build: cudaStreamSynchronize failed: ")
                                     + cudaGetErrorString(st));
        }

        cudaIpcMemHandle_t handle;
        st = cudaIpcGetMemHandle(&handle, buffer_);
        if (st != cudaSuccess) {
            throw std::runtime_error(std::string("NvIpcWriter::build: cudaIpcGetMemHandle failed: ")
                                     + cudaGetErrorString(st));
        }

        return py::bytes(reinterpret_cast<const char*>(&handle), sizeof(handle));
    }

    /**
     * @brief Get the number of bytes currently occupied in the buffer.
     *
     * @return std::size_t  The number of bytes written by write() calls.
     */
    std::size_t occupied_size() const {
        return occupied_size_;
    }

private:
    std::size_t  occupied_size_;
    std::size_t  buffer_size_;
    void*        buffer_;
    std::int32_t device_id_;
};

}  // namespace tipc