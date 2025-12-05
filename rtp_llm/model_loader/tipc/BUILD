package(default_visibility = ["//:__subpackages__"])

py_library(
    name = "tipc",
    data = [
        "csrc/common.h",
        "csrc/export.cc",
        "csrc/ipc.h",
        "csrc/shm.h",
        "csrc/bufferIO.h",
        "build/readme.md"
    ],
    srcs = glob([
        "*.py",
    ]),
)
