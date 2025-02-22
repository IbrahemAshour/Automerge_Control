"""
Model Predictive Controller (MPC) for autonomous vehicle trajectory tracking.
Uses CasADi for symbolic optimization and IPOPT as the solver.
"""

import numpy as np
import casadi as ca
from opencda.dynamics.vehicle_model import VehicleModel

class MPCController:
    def __init__(self, config):
        """
        Initialize MPC with configuration parameters.
        Args:
            config (dict): Configuration from YAML file.
        """
        # Prediction horizon and timestep
        self.horizon = config['horizon']  # Number of prediction steps
        self.dt = config['dt']  # Time per step (seconds)

        # Cost function weights
        self.Q = np.diag(config['Q'])  # State tracking weights [x, y, psi, vx, vy, psi_dot]
        self.R = np.diag(config['R'])  # Control smoothness weights [acc, delta]

        # Actuator limits
        self.a_max = config['a_max']  # Max acceleration (m/s²)
        self.delta_max = np.deg2rad(config['delta_max'])  # Max steering angle (radians)

        # Vehicle dynamics model
        self.dynamics = VehicleModel()
        # Setup MPC optimization problem
        self.setup_mpc()

    def setup_mpc(self):
        """Setup symbolic variables, cost function, and constraints for CasADi solver."""
        # Symbolic variables
        # X: States over horizon (6 x N)
        # U: Controls over horizon (2 x N-1)
        # X_ref: Reference trajectory (6 x N)
        self.X = ca.SX.sym('X', self.dynamics.n_states, self.horizon)
        self.U = ca.SX.sym('U', self.dynamics.n_controls, self.horizon - 1)
        self.X_ref = ca.SX.sym('X_ref', self.dynamics.n_states, self.horizon)

        # Initialize cost and constraints
        cost = 0  # Total cost
        constraints = []  # List of constraints

        # Constraint 1: Initial state must match current vehicle state
        constraints.append(self.X[:, 0] - self.X_ref[:, 0])

        # Dynamics constraints and cost calculation
        for t in range(self.horizon - 1):
            # Compute next state using the kinematic model
            state_t = self.X[:, t]
            control_t = self.U[:, t]
            next_state = state_t + self.dt * self.dynamics.kinematic_model(state_t, control_t)

            # Constraint 2: Next state must match the predicted state
            constraints.append(self.X[:, t + 1] - next_state)

            # Cost 1: Tracking error (deviation from reference trajectory)
            state_error = self.X[:, t] - self.X_ref[:, t]
            cost += ca.mtimes(state_error.T, ca.mtimes(self.Q, state_error))

            # Cost 2: Control smoothness (penalize abrupt changes)
            if t > 0:
                control_change = self.U[:, t] - self.U[:, t - 1]
                cost += ca.mtimes(control_change.T, ca.mtimes(self.R, control_change))

        # Cost 3: Terminal cost (prioritize final state accuracy)
        terminal_error = self.X[:, -1] - self.X_ref[:, -1]
        cost += 2 * ca.mtimes(terminal_error.T, ca.mtimes(self.Q, terminal_error))

        # Combine decision variables (flatten states and controls into a vector)
        opt_variables = ca.vertcat(
            ca.reshape(self.X, -1, 1),  # Flatten states into a column vector
            ca.reshape(self.U, -1, 1)   # Flatten controls into a column vector
        )

        # NLP problem setup
        nlp = {
            'x': opt_variables,  # Decision variables
            'f': cost,          # Objective function
            'g': ca.vertcat(*constraints),  # Constraints
            'p': ca.reshape(self.X_ref, -1, 1)  # Parameters (reference trajectory)
        }

        # Solver options (IPOPT with silent mode)
        opts = {
            'ipopt.print_level': 0,  # Suppress IPOPT output
            'print_time': 0,
            'ipopt.tol': 1e-8,       # Convergence tolerance
        }

        # Create solver
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    def update_info(self, ego_pos, ego_speed):
        """
        Update the controller with the vehicle's current state.
        Args:
            ego_pos (carla.Transform): Current position and orientation.
            ego_speed (float): Current longitudinal speed (m/s).
        """
        # Convert CARLA Transform to MPC state vector [x, y, psi, vx, vy, psi_dot]
        # Simplified: Assume vy=0 and psi_dot=0
        self.current_state = np.array([
            ego_pos.location.x,          # x position (m)
            ego_pos.location.y,          # y position (m)
            np.deg2rad(ego_pos.rotation.yaw),  # Yaw angle (radians)
            ego_speed,                   # Longitudinal speed (m/s)
            0,                           # Lateral speed (simplified)
            0                            # Yaw rate (simplified)
        ])

    def run_step(self, target_speed, waypoint):
        """
        Compute control commands for the current timestep.
        Args:
            target_speed (float): Desired speed (m/s).
            waypoint (list): List of CARLA waypoints for the reference trajectory.
        Returns:
            carla.VehicleControl: Throttle, steering, and brake commands.
        """
        # Generate reference trajectory from waypoints
        X_ref = self._generate_reference(waypoint)

        # Initial guess (warm-start with previous solution)
        initial_guess = np.concatenate([
            self.current_state.tolist() * self.horizon,  # Repeat current state for all steps
            self.last_controls.flatten()                 # Previous control sequence
        ])

        # Solve MPC problem
        sol = self.solver(
            x0=initial_guess,  # Initial guess
            p=X_ref.flatten(), # Reference trajectory (flattened)
            lbg=0,            # Lower bound for constraints (equalities)
            ubg=0,            # Upper bound for constraints (equalities)
            lbx=self._get_lower_bounds(),  # Control/state lower bounds
            ubx=self._get_upper_bounds()   # Control/state upper bounds
        )

        # Extract optimal controls
        opt_vars = sol['x'].full()
        controls = opt_vars[-self.dynamics.n_controls * (self.horizon - 1):]
        controls = controls.reshape(-1, self.dynamics.n_controls)
        acc, delta = controls[0]  # Use first control in the sequence

        # Save controls for next warm-start
        self.last_controls = controls

        # Convert to CARLA VehicleControl
        return self._to_carla_control(acc, delta)

    def _generate_reference(self, waypoints):
        """
        Generate reference trajectory from OpenCDA waypoints.
        Args:
            waypoints (list): List of CARLA waypoints.
        Returns:
            np.ndarray: Reference trajectory (6 x N)
        """
        # Simplified: Use the first N waypoints, assuming constant speed
        ref = []
        for wp in waypoints[:self.horizon]:
            ref.append([
                wp.transform.location.x,
                wp.transform.location.y,
                np.deg2rad(wp.transform.rotation.yaw),
                target_speed,
                0,  # Lateral speed
                0   # Yaw rate
            ])
        return np.array(ref).T  # Transpose to (6 x N)

    def _get_lower_bounds(self):
        """Lower bounds for states and controls."""
        # Control bounds
        lb_controls = [-self.a_max, -self.delta_max] * (self.horizon - 1)
        # State bounds (loose bounds for x, y, yaw; vx >= 0)
        lb_states = [-np.inf, -np.inf, -np.inf, 0, -np.inf, -np.inf] * self.horizon
        return lb_controls + lb_states

    def _get_upper_bounds(self):
        """Upper bounds for states and controls."""
        # Control bounds
        ub_controls = [self.a_max, self.delta_max] * (self.horizon - 1)
        # State bounds (loose bounds for x, y, yaw; vx <= 60 m/s)
        ub_states = [np.inf, np.inf, np.inf, 60, np.inf, np.inf] * self.horizon
        return ub_controls + ub_states

    def _to_carla_control(self, acc, delta):
        """
        Convert MPC outputs to CARLA VehicleControl.
        Args:
            acc (float): Acceleration command (m/s²)
            delta (float): Steering angle (radians)
        Returns:
            carla.VehicleControl: Normalized throttle, steer, and brake.
        """
        control = carla.VehicleControl()
        # Normalize acceleration to throttle (0-1) or brake (0-1)
        control.throttle = np.clip(acc / self.a_max, 0, 1)
        control.steer = np.clip(delta / self.delta_max, -1, 1)
        control.brake = 1.0 if acc < 0 else 0.0  # Full brake if decelerating
        return control