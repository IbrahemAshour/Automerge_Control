"""
Defines the vehicle's kinematic/dynamic model for MPC.
States: [x, y, yaw, longitudinal velocity (vx), lateral velocity (vy), yaw rate (psi_dot)]
Controls: [acceleration (acc), steering angle (delta)]
"""

import casadi as ca
import numpy as np

class VehicleModel:
    def __init__(self):
        # Number of states and controls
        self.n_states = 6  # [x, y, yaw, vx, vy, psi_dot]
        self.n_controls = 2  # [acc, delta]
        self.L = 2.5  # Wheelbase (meters) - distance between front and rear axles

    def kinematic_model(self, state, control):
        """
        Kinematic bicycle model (simplified for real-time computation).
        Args:
            state (list/casadi.SX): Current state vector [x, y, psi, vx, vy, psi_dot]
            control (list/casadi.SX): Control inputs [acc, delta]
        Returns:
            casadi.SX: State derivatives (dx/dt, dy/dt, etc.)
        """
        x, y, psi, vx, vy, psi_dot = state
        acc, delta = control

        # Equations of motion
        dx = vx * ca.cos(psi) - vy * ca.sin(psi)  # x velocity in global frame
        dy = vx * ca.sin(psi) + vy * ca.cos(psi)  # y velocity in global frame
        dpsi = psi_dot  # Yaw rate
        dvx = acc  # Longitudinal acceleration
        dvy = 0  # Simplified: Assume lateral velocity is negligible
        dpsi_dot = (vx / self.L) * ca.tan(delta)  # Yaw acceleration from steering

        return ca.vertcat(dx, dy, dpsi, dvx, dvy, dpsi_dot)