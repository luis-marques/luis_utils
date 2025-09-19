"""Algorithm definitions and configuration mapping utilities."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, Tuple

from pymatlie.se2 import SE2

from luis_utils.load import instantiate_class
from luis_utils.systems import (
    second_order_unicycle,
)


class RobotType(Enum):
    """Enum for robot types."""

    UNICYCLE = auto()


class Q_Space(Enum):
    """Enum for configuration space types."""

    STATE_SPACE = auto()
    LIE = auto()


class NonconformityScore(Enum):
    """Enum for nonconformity score types."""

    MAHALANOBIS = auto()
    L2 = auto()


# Define unified robot system registry with class and factory
ROBOT_SYSTEMS: Dict[Tuple[RobotType, Q_Space], Dict[str, Any]] = {
    (RobotType.UNICYCLE, Q_Space.STATE_SPACE): {
        "cls": second_order_unicycle.Unicycle2ndOrder,
        "factory": lambda inertia_matrix: second_order_unicycle.Unicycle2ndOrder(inertia_matrix=inertia_matrix),
    },
    (RobotType.UNICYCLE, Q_Space.LIE): {
        "cls": second_order_unicycle.SE2Unicycle2ndOrder,
        "factory": lambda inertia_matrix: second_order_unicycle.SE2Unicycle2ndOrder(inertia_matrix=inertia_matrix),  # type: ignore[no-untyped-call]
    },
}

ROBOT_TO_KEY = {entry["cls"]: key for key, entry in ROBOT_SYSTEMS.items()}

ROBOT_QSPACE_TO_FACTORY = {key: val["factory"] for key, val in ROBOT_SYSTEMS.items()}
ROBOT_QSPACE_TO_CLASS = {key: val["cls"] for key, val in ROBOT_SYSTEMS.items()}

ALL_SYSTEM_CLASSES = set(entry["cls"] for entry in ROBOT_SYSTEMS.values())  # HARMONIC_OSCILLATOR, UNICYCLE2NDORDER, SE2, ...


@dataclass  # (frozen=True)
class Algorithm:
    """Algorithm configuration and planner instantiation."""

    name: str
    planner_name: str
    model_system: object

    initial_condition: Dict
    sensor: Optional[object] = None

    planner: object = field(init=False)
    planner_kwargs: Dict[str, Any] = field(default_factory=dict)

    # config_space: Q_Space = field(init=False)
    def __post_init__(self):
        if self.planner_name == "fixed_action":
            self.planner = instantiate_class(
                module_name="luis_utils.planners.fixed_action",
                class_name="FixedActionPlanner",
                class_args={"system": self.model_system, **self.planner_kwargs},
            )
        else:
            raise ValueError(f"Planner {self.planner_name} not found")


@dataclass(frozen=True, slots=True)
class AlgorithmCfg:
    """Algorithm configuration parameters."""

    name: str
    plan_with_obs: bool
    planner: str
    sensor: str
    cp_algo: Optional[str] = None
    belief_model: Optional[str] = None


class ConfigMapper:
    """Configuration space mapping registry."""

    _registry: Dict[Tuple[Tuple[RobotType, Q_Space], Tuple[RobotType, Q_Space]], Callable[[Any], Any]] = {}

    @staticmethod
    def register(from_key: Tuple[RobotType, Q_Space], to_key: Tuple[RobotType, Q_Space], fn: Callable[[Any], Any]) -> None:
        """Register a configuration mapping function."""
        ConfigMapper._registry[(from_key, to_key)] = staticmethod(fn)

    @staticmethod
    def map(from_key: Tuple[RobotType, Q_Space], to_key: Tuple[RobotType, Q_Space], q: Any) -> Any:
        """Apply configuration mapping."""
        lookup_key = (from_key, to_key)
        return ConfigMapper._registry[lookup_key](q)


class VelocityMapper:
    """Velocity space mapping registry."""

    _registry: Dict[Tuple[Tuple[RobotType, Q_Space], Tuple[RobotType, Q_Space]], Callable[[Any, Any], Any]] = {}

    @staticmethod
    def register(from_key: Tuple[RobotType, Q_Space], to_key: Tuple[RobotType, Q_Space], fn: Callable[[Any, Any], Any]) -> None:
        """Register a velocity mapping function."""
        VelocityMapper._registry[(from_key, to_key)] = staticmethod(fn)

    @staticmethod
    def map(from_key: Tuple[RobotType, Q_Space], to_key: Tuple[RobotType, Q_Space], q: Any, dq_or_v: Any) -> Any:
        """Apply velocity mapping."""
        lookup_key = (from_key, to_key)
        return VelocityMapper._registry[lookup_key](q, dq_or_v)


# Identity mappings
for robot_type in RobotType:
    for space in Q_Space:
        key = (robot_type, space)
        ConfigMapper.register(key, key, lambda q: q)
        VelocityMapper.register(key, key, lambda q, dq: dq)

# Unicycle-specific mappings (Euclidean ↔ Lie)
for r_type in [RobotType.UNICYCLE]:  # SE2 Robot types
    ConfigMapper.register((r_type, Q_Space.STATE_SPACE), (r_type, Q_Space.LIE), SE2.map_q_to_configuration)
    ConfigMapper.register((r_type, Q_Space.LIE), (r_type, Q_Space.STATE_SPACE), SE2.map_configuration_to_q)
    VelocityMapper.register((r_type, Q_Space.STATE_SPACE), (r_type, Q_Space.LIE), SE2.map_dq_to_velocity)
    VelocityMapper.register((r_type, Q_Space.LIE), (r_type, Q_Space.STATE_SPACE), SE2.map_velocity_to_dq)
