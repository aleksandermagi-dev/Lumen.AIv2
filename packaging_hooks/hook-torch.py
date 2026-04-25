from __future__ import annotations

from PyInstaller import compat
from PyInstaller.utils.hooks import PY_DYLIB_PATTERNS, collect_data_files, collect_dynamic_libs, collect_submodules

module_collection_mode = "pyz+py"
warn_on_missing_hiddenimports = False

_EXCLUDED_PREFIXES = (
    "torch._dynamo.backends.distributed",
    "torch._inductor",
    "torch.contrib._tensorboard_vis",
    "torch.distributed",
    "torch.onnx",
    "torch.testing",
    "torch.utils.tensorboard",
)


def _keep_torch_module(name: str) -> bool:
    return not any(name == prefix or name.startswith(f"{prefix}.") for prefix in _EXCLUDED_PREFIXES)


datas = collect_data_files(
    "torch",
    excludes=[
        "**/*.h",
        "**/*.hpp",
        "**/*.cuh",
        "**/*.lib",
        "**/*.cpp",
        "**/*.pyi",
        "**/*.cmake",
        "**/test/**",
        "**/tests/**",
    ],
)
hiddenimports = collect_submodules("torch", filter=_keep_torch_module, on_error="ignore")
binaries = collect_dynamic_libs("torch", search_patterns=PY_DYLIB_PATTERNS + ["*.so.*"])

if compat.is_linux:
    bindepend_symlink_suppression = ["**/torch/lib/*.so*"]
