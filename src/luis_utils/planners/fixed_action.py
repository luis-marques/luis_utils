"""A planner that simply replays a fixed, precomputed sequence."""

from luis_utils.planners.base import BasePlanner


class FixedActionPlanner(BasePlanner):
    """A planner that simply replays a fixed, precomputed sequence."""

    def __init__(self, system, planned_controls):
        """A planner that simply replays a fixed, precomputed sequence.

        Args:
          system:       your dynamics object (used only to get u_dim).
          planned_controls:  a (H, u_dim) tensor of controls.
        """
        super().__init__(system, planned_controls.shape[0])
        self._fixed_control_sequence = planned_controls.clone()
        assert self._fixed_control_sequence.ndim == 4, "Fixed control sequence must be (H, N_EP, A_DIM, N_EGOS)"

    def plan(self, obs: dict, ep_index: int):
        if self.planned_controls is None:
            self.planned_controls = self._fixed_control_sequence[:, ep_index].clone()
