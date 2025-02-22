"""
Integrates the MPC controller into OpenCDA's control loop.
"""

class ControlManager:
    def __init__(self, control_config):
        """
        Initialize the MPC controller.
        Args:
            control_config (dict): Configuration from YAML.
        """
        from opencda.core.actuation.mpc_controller import MPCController
        # Initialize MPC with parameters from YAML
        self.controller = MPCController(control_config['args'])

    def update_info(self, ego_pos, ego_speed):
        """
        Update the controller with the vehicle's current state.
        Args:
            ego_pos (carla.Transform): Current position and orientation.
            ego_speed (float): Current speed (m/s).
        """
        self.controller.update_info(ego_pos, ego_speed)

    def run_step(self, target_speed, waypoint):
        """
        Execute one control step.
        Args:
            target_speed (float): Desired speed (m/s).
            waypoint (list): Reference trajectory waypoints.
        Returns:
            carla.VehicleControl: Throttle, steer, and brake commands.
        """
        return self.controller.run_step(target_speed, waypoint)