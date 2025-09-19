"""Tests for the load module."""

import sys
import types
from typing import Callable

import pytest

from luis_utils.load import get_callable, get_class


# Creating fake modules for test functions in file
@pytest.fixture(scope="module", autouse=True)
def mock_modules():
    """Create mock modules for testing."""
    # Module with a simple function
    mod1 = types.ModuleType("mod1")

    def test_func():
        return "test_func"

    mod1.test_func = test_func

    # Module with a class and method
    mod2 = types.ModuleType("mod2")

    class TestClass:
        """Test class."""

        @staticmethod
        def test_method():
            """Test method."""
            return "test_method"

    mod2.TestClass = TestClass

    # Module with a non-callable
    mod3 = types.ModuleType("mod3")
    mod3.value = 123

    sys.modules["mod1"] = mod1
    sys.modules["mod2"] = mod2
    sys.modules["mod3"] = mod3

    yield

    # Cleanup
    del sys.modules["mod1"]
    del sys.modules["mod2"]
    del sys.modules["mod3"]


# --------------------------
# Tests for get_callable function
# --------------------------
def test_function_lookup():
    """Test looking up a simple function."""
    fn = get_callable("mod1", "test_func")
    assert isinstance(fn, Callable)
    assert fn() == "test_func"


def test_class_method_lookup():
    """Test looking up a class method."""
    method = get_callable("mod2", "TestClass.test_method")
    assert isinstance(method, Callable)
    assert method() == "test_method"


def test_invalid_dot_format():
    """Test error on invalid dot format."""
    with pytest.raises(AssertionError, match="Only one dot"):
        get_callable("mod2", "A.B.C")


def test_missing_function():
    """Test error when function is missing."""
    with pytest.raises(ImportError, match="not found in 'mod1'"):
        get_callable("mod1", "missing_fn")


def test_missing_class():
    """Test error when class is missing."""
    with pytest.raises(AttributeError, match="Class MissingClass not found"):
        get_callable("mod2", "MissingClass.test_method")


def test_non_callable_attribute():
    """Test error when attribute is not callable."""
    with pytest.raises(TypeError, match="'value' in 'mod3' is not callable"):
        get_callable("mod3", "value")


def test_missing_module():
    """Test error when module is missing."""
    with pytest.raises(ImportError, match="Module nonexistent_module not found"):
        get_callable("nonexistent_module", "test_func")


# --------------------------
# Tests for get_class function
# --------------------------
def test_get_existing_class_case_insensitive():
    """Test getting class with case insensitive lookup."""
    cls = get_class("mod2", "testclass")  # lowercase
    assert cls.__name__ == "TestClass"


def test_get_existing_class_exact_case():
    """Test getting class with exact case match."""
    cls = get_class("mod2", "TestClass")
    assert cls.__name__ == "TestClass"


def test_get_missing_class():
    """Test error when getting missing class."""
    with pytest.raises(AttributeError, match="Class 'MissingClass' not found in module 'mod2'"):
        get_class("mod2", "MissingClass")


def test_get_class_from_missing_module():
    """Test error when getting class from missing module."""
    with pytest.raises(ImportError, match="Module 'nonexistent' could not be imported"):
        get_class("nonexistent", "Foo")


def test_get_class_from_module_with_no_classes():
    """Test error when getting class from module with no classes."""
    with pytest.raises(AttributeError, match="Class 'Value' not found in module 'mod3'"):
        get_class("mod3", "Value")
