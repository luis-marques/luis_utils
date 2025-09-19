"""Module loading and utility functions."""

import cProfile
import datetime
import importlib
import inspect
import sys
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import plotly.io as pio
import torch


def get_callable(module_name: str, callable_name: str) -> Callable:
    """Returns callable function or method from the module listed.

    callable_name can either be: "function_name" or "class_name.method_name"
    """
    if "." in callable_name:
        assert callable_name.count(".") == 1, "Only one dot is allowed in callable_name"
        class_name, function_name = callable_name.split(".")
    else:
        class_name, function_name = None, callable_name

    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Module {module_name} not found. Error: {e}")

    obj = module
    if class_name:
        try:
            obj = getattr(module, class_name)
        except AttributeError as e:
            raise AttributeError(f"Class {class_name} not found in module {module_name}. Error: {e}")

    try:
        candidate = getattr(obj, function_name)
    except AttributeError:
        raise ImportError(f"Function or method '{function_name}' not found in '{module_name}'")

    if not callable(candidate):
        raise TypeError(f"'{function_name}' in '{module_name}' is not callable")

    return candidate


def get_class(module_name: str, class_name: str) -> type:
    """Returns class (case insensitive) from module."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Module '{module_name}' could not be imported. Error: {e}")

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if name.lower() == class_name.lower():
            return obj

    raise AttributeError(f"Class '{class_name}' not found in module '{module_name}'")


def instantiate_class(module_name: str, class_name: str, class_args: dict):
    """Returns instance of class from module."""
    try:
        _class = get_class(module_name, class_name)
        return _class(**class_args)
    except Exception as e:
        raise AttributeError(f"Error instantiating class {class_name} from module {module_name}. Error: {e}")


def get_ABC_implementations(module_name: str, ABC_name: str) -> List[str]:
    """Returns list of classes that implement the given ABC."""
    _module = importlib.import_module(f"{module_name}")
    _ABC = getattr(_module, f"{ABC_name}")
    return [name for name, obj in inspect.getmembers(_module) if inspect.isclass(obj) and _ABC in obj.__bases__]


def get_repo_path(path: Optional[Path] = None) -> Path:
    """Get the path to the repository root."""
    if path is None:
        caller_frame = inspect.stack()[1]
        path = Path(caller_frame.filename)

    if path is None:
        raise ValueError("Unable to determine path from caller frame")

    start_path = Path(path).resolve()
    # Walk up the directory tree from the starting path
    for parent in [start_path] + list(start_path.parents):
        git_dir = parent / ".git"
        if git_dir.is_dir():
            return parent

    raise FileNotFoundError(f"Repository root not found starting from {start_path}")


def get_torch_device() -> torch.device:
    """Get available torch device (gpu takes priority over cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def startup_torch(torch_device=None, torch_precision: str = "double") -> None:
    """Set up PyTorch environment."""
    if torch_device is None:
        torch_dev = get_torch_device()
    else:
        torch_dev = torch_device
    torch.set_default_device(torch_dev)
    if torch_precision == "double":
        if torch_dev == torch.device("cuda"):
            torch.set_default_tensor_type("torch.cuda.DoubleTensor")
        else:
            torch.set_default_tensor_type("torch.DoubleTensor")
        torch.set_default_dtype(torch.float64)
    elif torch_precision == "single":
        if torch_dev == torch.device("cuda"):
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")
        torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def script_starter(precision: str, profile: bool = False, device=None) -> None:
    """Set up the PyTorch environment."""

    if "torch" in sys.modules:
        sys.modules["torch"] = torch  # Lock torch version
        startup_torch(torch_device=device, torch_precision=precision)

    if "numpy" in sys.modules:
        np.random.seed(0)

    if "plotly" in sys.modules:

        pio.renderers.default = "iframe"

    if profile:
        profiler = cProfile.Profile()
        profiler.enable()
        prof_path = f"profile_{datetime.datetime.now():%Y%m%d_%H%M%S}.prof"
        print("[profiler] enabled, will dump to", prof_path)
