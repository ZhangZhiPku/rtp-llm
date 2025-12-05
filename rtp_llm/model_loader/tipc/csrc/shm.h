#pragma once

#include "common.h"
#include "bufferIO.h"

#include <string>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace tipc {

/**
 * @brief Reader for shared-memory (shm) files into CUDA tensors.
 *
 * NvShmReader maps a shm-backed file into the process address space
 * and provides an asynchronous host-to-device copy interface to read
 * slices of the shm region into CUDA tensors.
 */
class NvShmReader: public BufferedTensorReader {
public:
    /**
     * @brief Construct a new NvShmReader from a shm-backed file.
     *
     * @param shm_file_name Path to the shm file (e.g. "/dev/shm/xxx" or a file
     *        created via shm_open).
     *
     * This constructor opens the file, queries its size, and mmaps it into the
     * process address space with read-only protection.
     */
    explicit NvShmReader(const std::string& shm_file_name): shm_(shm_file_name), fd_(-1), ptr_(nullptr), size_(0) {
        fd_ = ::open(shm_file_name.c_str(), O_RDONLY);
        TORCH_CHECK(fd_ >= 0, "NvShmReader: failed to open shm file: ", shm_file_name);

        struct stat st{};
        const int   ret = ::fstat(fd_, &st);
        TORCH_CHECK(ret == 0, "NvShmReader: fstat failed for shm file: ", shm_file_name);
        TORCH_CHECK(st.st_size > 0, "NvShmReader: shm file size is 0: ", shm_file_name);
        size_ = static_cast<std::size_t>(st.st_size);

        void* addr = ::mmap(
            /*addr=*/nullptr,
            /*length=*/size_,
            /*prot=*/PROT_READ,
            /*flags=*/MAP_SHARED,
            /*fd=*/fd_,
            /*offset=*/0);
        TORCH_CHECK(addr != MAP_FAILED, "NvShmReader: mmap failed for shm file: ", shm_file_name);

        ptr_ = addr;
    }

    /**
     * @brief Destroy the NvShmReader object.
     *
     * Unmaps the shm region and closes the file descriptor if they are valid.
     */
    ~NvShmReader() override {
        try {
            close();
        } catch (...) {
            // do not throw from destructor
        }
    }

    NvShmReader(const NvShmReader&)            = delete;
    NvShmReader& operator=(const NvShmReader&) = delete;

    /**
     * @brief Implementation of BufferedTensorReader::read().
     *
     * Read slices from the shm buffer into 1D uint8 CUDA tensors.
     *
     * @param total_bytes Total valid bytes in the shm region. Must be > 0 and
     *                    less than or equal to the mapped region size().
     * @param offsets     Starting offsets (in bytes) of each slice. Each slice i
     *                    is [offsets[i], offsets[i+1]) except the last slice,
     *                    which is [offsets.back(), total_bytes).
     * @param device_id   CUDA device on which the returned tensors should reside.
     *
     * @return std::vector<torch::Tensor>
     *         A list of 1D tensors of dtype torch::kUInt8, each tensor storing
     *         a copy of the corresponding slice from shm on the given device.
     *
     * Note:
     *  - This implementation performs asynchronous HostToDevice copies and
     *    returns owning tensors (no view).
     */
    std::vector<torch::Tensor>
    read(std::size_t total_bytes, const std::vector<std::int64_t>& offsets, std::int32_t device_id) override {
        TORCH_CHECK(total_bytes > 0, "NvShmReader::read: total_bytes must be positive");
        TORCH_CHECK(total_bytes <= size_,
                    "NvShmReader::read: total_bytes exceeds shm region size (",
                    total_bytes,
                    " > ",
                    size_,
                    ")");

        if (offsets.empty()) {
            return {};
        }

        cudaError_t st = cudaSetDevice(device_id);
        if (st != cudaSuccess) {
            throw std::runtime_error(std::string("NvShmReader::read: cudaSetDevice failed: ") + cudaGetErrorString(st));
        }

        const auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device_id);

        std::vector<torch::Tensor> result;
        result.reserve(offsets.size());

        for (std::size_t i = 0; i < offsets.size(); ++i) {
            std::int64_t start = offsets[i];
            TORCH_CHECK(start >= 0 && static_cast<std::size_t>(start) < total_bytes,
                        "NvShmReader::read: offset out of range: ",
                        start);

            std::int64_t end = (i + 1 < offsets.size()) ? offsets[i + 1] : static_cast<std::int64_t>(total_bytes);
            TORCH_CHECK(end > start && static_cast<std::size_t>(end) <= total_bytes,
                        "NvShmReader::read: invalid offset range: [",
                        start,
                        ", ",
                        end,
                        ")");

            std::size_t len     = static_cast<std::size_t>(end - start);
            const void* src_ptr = static_cast<const char*>(ptr_) + start;

            // Allocate destination tensor on the target device.
            torch::Tensor dst     = torch::empty({static_cast<long>(len)}, opts);
            void*         dst_ptr = dst.data_ptr();

            auto         stream      = at::cuda::getCurrentCUDAStream(device_id);
            cudaStream_t cuda_stream = stream.stream();

            const cudaError_t err = cudaMemcpyAsync(dst_ptr, src_ptr, len, cudaMemcpyHostToDevice, cuda_stream);
            TORCH_CHECK(err == cudaSuccess,
                        "NvShmReader::read: cudaMemcpyAsync(HostToDevice) failed: ",
                        cudaGetErrorString(err));

            result.push_back(dst);
        }

        return result;
    }

    /**
     * @brief Close the shm mapping and file descriptor.
     *
     * After close() is called, further read() calls are invalid.
     */
    void close() override {
        if (ptr_ && ptr_ != MAP_FAILED && size_ > 0) {
            ::munmap(ptr_, size_);
            ptr_  = nullptr;
            size_ = 0;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

private:
    std::string shm_;   ///< Path to the shm file (for logging/debugging).
    int         fd_;    ///< File descriptor of the shm file.
    void*       ptr_;   ///< Base address of the mapped shm region.
    std::size_t size_;  ///< Size of the mapped shm region in bytes.
};

/**
 * @brief Writer for shared-memory (shm) files from CUDA tensors.
 *
 * NvShmWriter maps a shm-backed file into the process address space
 * with write permission and provides an asynchronous device-to-host
 * copy interface to write CUDA tensors into the shm region.
 *
 * It maintains an internal write offset that is advanced on each write().
 */
class NvShmWriter: public BufferedTensorWriter {
public:
    /**
     * @brief Construct a new NvShmWriter from a shm-backed file.
     *
     * @param shm_file_name Path to the shm file (e.g. "/dev/shm/xxx" or a file
     *        created via shm_open).
     *
     * This constructor opens the file, queries its size, and mmaps it into the
     * process address space with read/write protection.
     */
    explicit NvShmWriter(const std::string& shm_file_name):
        shm_(shm_file_name), fd_(-1), ptr_(nullptr), size_(0), write_offset_(0) {
        fd_ = ::open(shm_file_name.c_str(), O_RDWR);
        TORCH_CHECK(fd_ >= 0, "NvShmWriter: failed to open shm file: ", shm_file_name);

        struct stat st{};
        const int   ret = ::fstat(fd_, &st);
        TORCH_CHECK(ret == 0, "NvShmWriter: fstat failed for shm file: ", shm_file_name);
        TORCH_CHECK(st.st_size > 0, "NvShmWriter: shm file size is 0: ", shm_file_name);
        size_ = static_cast<std::size_t>(st.st_size);

        void* addr = ::mmap(
            /*addr=*/nullptr,
            /*length=*/size_,
            /*prot=*/PROT_READ | PROT_WRITE,
            /*flags=*/MAP_SHARED,
            /*fd=*/fd_,
            /*offset=*/0);
        TORCH_CHECK(addr != MAP_FAILED, "NvShmWriter: mmap failed for shm file: ", shm_file_name);

        ptr_ = addr;
    }

    /**
     * @brief Destroy the NvShmWriter object.
     *
     * Unmaps the shm region and closes the file descriptor if they are valid.
     */
    ~NvShmWriter() override {
        try {
            close();
        } catch (...) {
            // do not throw from destructor
        }
    }

    NvShmWriter(const NvShmWriter&)            = delete;
    NvShmWriter& operator=(const NvShmWriter&) = delete;

    /**
     * @brief Implementation of BufferedTensorWriter::write().
     *
     * Asynchronously copy data from a CUDA tensor to the shm region at the
     * current write offset, then advance the offset.
     *
     * @param t  Source tensor on a CUDA device. Must be contiguous.
     *
     * @return std::size_t
     *         Number of bytes written, or 0 if there is not enough space.
     *
     * The copy is enqueued on the current PyTorch CUDA stream associated with
     * the tensor's device using cudaMemcpyAsync with cudaMemcpyDeviceToHost.
     * Synchronization is the caller's responsibility.
     */
    std::size_t write(const torch::Tensor& t) override {
        TORCH_CHECK(t.is_cuda(), "NvShmWriter::write: tensor must be on CUDA device");
        TORCH_CHECK(t.is_contiguous(), "NvShmWriter::write: tensor must be contiguous");

        std::size_t bytes = static_cast<std::size_t>(t.nbytes());
        if (write_offset_ + bytes > size_) {
            return 0;
        }

        void*       dst_ptr = static_cast<char*>(ptr_) + write_offset_;
        const void* src_ptr = t.data_ptr();

        auto         stream      = at::cuda::getCurrentCUDAStream(t.device().index());
        cudaStream_t cuda_stream = stream.stream();

        const cudaError_t err = cudaMemcpyAsync(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToHost, cuda_stream);
        TORCH_CHECK(
            err == cudaSuccess, "NvShmWriter::write: cudaMemcpyAsync(DeviceToHost) failed: ", cudaGetErrorString(err));

        write_offset_ += bytes;
        return bytes;
    }

    /**
     * @brief Reset the internal write offset to the beginning of the shm region.
     *
     * After reset(), the next write() call starts at offset 0.
     * The underlying memory is not cleared.
     */
    void reset() override {
        write_offset_ = 0;
    }

    /**
     * @brief Close the shm mapping and file descriptor.
     *
     * After close(), further write() calls are invalid.
     */
    void close() override {
        if (ptr_ && ptr_ != MAP_FAILED && size_ > 0) {
            ::munmap(ptr_, size_);
            ptr_          = nullptr;
            size_         = 0;
            write_offset_ = 0;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

private:
    std::string shm_;           ///< Path to the shm file (for logging/debugging).
    int         fd_;            ///< File descriptor of the shm file.
    void*       ptr_;           ///< Base address of the mapped shm region.
    std::size_t size_;          ///< Size of the mapped shm region in bytes.
    std::size_t write_offset_;  ///< Current write offset in bytes.
};

}  // namespace tipc