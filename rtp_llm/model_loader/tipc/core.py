import base64
import json
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch

COMMON_PREFIX = "TIPC_TRANSPROTING"
MethodType = Literal["shm", "cuipc"]


def torch_dtype_to_str(dtype: torch.dtype) -> str:
    """Converts a PyTorch dtype to its string representation."""
    return str(dtype).replace("torch.", "")


def str_to_torch_dtype(dtype: str) -> torch.dtype:
    """Converts a string representation of a PyTorch dtype back to its corresponding dtype object."""
    return getattr(torch, dtype)


def np_dtype_to_str(dtype: np.dtype) -> str:
    """Converts a NumPy dtype to its string representation."""
    return str(dtype)


def str_to_np_dtype(dtype_str: str) -> np.dtype:
    """Converts a string representation of a NumPy dtype back to its corresponding dtype object."""
    return np.dtype(dtype_str)


@dataclass
class TensorIPCMeta:
    """Data class representing the metadata required to rebuild a torch.Tensor"""

    name: str  # tensor name
    shape: torch.Size
    dtype: (
        torch.dtype
    )  # PyTorch dtype, will be converted to NumPy dtype for shared memory operations
    size: int  # Total size of the tensor data within the shared memory block, in bytes

    @classmethod
    def decode(cls, encoded: str) -> "TensorIPCMeta":
        """Decodes a base64 string back into a SharedMemIpcMeta instance."""
        decoded_bytes = base64.b64decode(encoded)
        serialized_dict = json.loads(decoded_bytes.decode("utf-8"))

        # Convert string representations back to original types
        serialized_dict["shape"] = torch.Size(serialized_dict["shape"])
        serialized_dict["dtype"] = str_to_torch_dtype(serialized_dict["dtype"])
        # Stride is already tuple, no conversion needed.

        return cls(**serialized_dict)

    def encode(self) -> str:
        """Encodes this SharedMemIpcMeta instance into a base64 string."""
        metadata_dict = asdict(self)
        # Convert specific types to serializable formats
        metadata_dict["shape"] = tuple(self.shape)
        metadata_dict["dtype"] = torch_dtype_to_str(self.dtype)

        json_string = json.dumps(metadata_dict)
        return base64.b64encode(json_string.encode("utf-8")).decode("utf-8")
