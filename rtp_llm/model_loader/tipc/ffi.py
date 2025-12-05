import os
from typing import List

import torch
from torch.utils.cpp_extension import CUDAExtension, load

from ._c import NvIpcReader, NvIpcWriter, NvShmReader, NvShmWriter

SOURCE_DIR = "tipc/csrc"
EXTENSION_BUILD_DIR = "tipc/build"
PROGRAM_NAME = "tipc"


class __CompileHelper__:
    def __init__(self) -> None:
        self.BUILD_DIR = EXTENSION_BUILD_DIR
        self.__CUDA_EXTENTION__ = None

        if torch.__version__ < "1.6.0":
            raise RuntimeError(
                f"{PROGRAM_NAME} cannot finish compile; PyTorch version 1.6 or higher is required."
            )

    def compile(self) -> CUDAExtension:
        """
        Compiles a CUDA extension from all source files in the source directory.

        This function automatically finds all .c, .cc, .cpp, and .cu files
        in the specified source directory and compiles them using JIT compilation.
        The compiled extension is a CUDAExtension object that can be called from Python.

        Requires CUDA and C++17 for compilation.
        """
        print(
            f"{PROGRAM_NAME} is currently compiling the code, which may take some time. "
            f"If any errors occur, please check your compilation environment: {PROGRAM_NAME} "
            "requires C++17 and CUDA."
        )

        # delete lock file.
        lock_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), self.BUILD_DIR, "lock"
        )
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except Exception as e:
                raise PermissionError(
                    f"Can not delete lock file at {lock_file}, delete it first!"
                )

        sources = self._find_all_source_files(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), SOURCE_DIR)
        )

        self.__CUDA_EXTENTION__ = load(
            name=PROGRAM_NAME,
            sources=sources,
            extra_include_paths=[
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "csrc"),
            ],
            build_directory=os.path.join(
                os.path.dirname(os.path.dirname(__file__)), self.BUILD_DIR
            ),
            with_cuda=True,
            extra_cuda_cflags=["-O3", "-use_fast_math"],
            extra_cflags=["-O3"],
        )
        return self.__CUDA_EXTENTION__

    def _find_all_source_files(self, directory: str) -> List[str]:
        """Recursively finds all C/C++ and CUDA source files in a directory."""
        source_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith((".c", ".cc", ".cpp", ".cu")):
                    source_files.append(os.path.join(root, file))
        return source_files

    @property
    def CUDA_EXTENSION(self):
        if self.__CUDA_EXTENTION__ is None:
            self.compile()
        return self.__CUDA_EXTENTION__


CompileHelper = __CompileHelper__()


def CONTIGUOUS_TENSOR(tensor: torch.Tensor):
    """Helper function"""
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class TipcLib:
    """Helper class for calling Compiled Methods."""

    @staticmethod
    def NvIpcWriter(device: int, buffer_size: int) -> NvIpcWriter:
        return CompileHelper.CUDA_EXTENSION.NvIpcWriter(device, buffer_size)

    @staticmethod
    def NvIpcReader(handle: bytes) -> NvIpcReader:
        return CompileHelper.CUDA_EXTENSION.NvIpcReader(handle)

    @staticmethod
    def NvShmWriter(shm_file_path: str) -> NvShmWriter:
        if shm_file_path.startswith("/dev/shm/"):
            return CompileHelper.CUDA_EXTENSION.NvShmWriter(shm_file_path)
        else:
            raise ValueError(
                f"NvShmWriter error, 不能打开给定的文件，因为 shm_file_path: {shm_file_path} 似乎不是一个可以被打开的 shared memory 文件"
            )

    @staticmethod
    def NvShmReader(shm_file_path: str) -> NvShmReader:
        if shm_file_path.startswith("/dev/shm/"):
            return CompileHelper.CUDA_EXTENSION.NvShmReader(shm_file_path)
        else:
            raise ValueError(
                f"NvShmReader error, 不能打开给定的文件，因为 shm_file_path: {shm_file_path} 似乎不是一个可以被打开的 shared memory 文件"
            )


__all__ = ["TipcLib", "NvShmReader", "NvShmWriter", "NvIpcWriter", "NvIpcReader"]
