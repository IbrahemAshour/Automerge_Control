"""
Test script to run the MPC controller in OpenCDA.
"""

import carla
import time
from opencda.core.common.cav_world import CavWorld
from opencda.scenario_testing.sim_api.sim_api import SimAPI
from opencda.scenario_testing.config_yaml import load_yaml

def run_scenario():
    """
    Main function to run the MPC-controlled vehicle in CARLA.
    """
    # -------------------------------------------------------------------------
    # 1. Load Configuration and Initialize OpenCDA
    # -------------------------------------------------------------------------
    # Load the YAML configuration file for MPC
    scenario_params = load_yaml('scenarios/mpc_test.yaml')
    
    # Create the CAV (Connected and Automated Vehicle) world 
    # (manages all vehicles/controllers)
    cav_world = CavWorld()

    # Initialize the simulation API (handles CARLA connection, spawning, etc.)
    sim_api = SimAPI(scenario_params, cav_world=cav_world)

    # -------------------------------------------------------------------------
    # 2. Spawn the Ego Vehicle with MPC Controller
    # -------------------------------------------------------------------------
    # Spawn the ego vehicle at default location (you can customize this)
    # The controller is automatically initialized based on the YAML config
    ego_vehicle = sim_api.spawn_vehicle(vehicle_type='default')

    # -------------------------------------------------------------------------
    # 3. Set Destination for the Ego Vehicle
    # -------------------------------------------------------------------------
    # Define a destination 200 meters ahead on the same lane
    destination = carla.Location(x=200, y=0, z=0)
    
    # Use OpenCDA's route planner to generate waypoints to the destination
    sim_api.set_destination(ego_vehicle, destination)

    # -------------------------------------------------------------------------
    # 4. Main Simulation Loop
    # -------------------------------------------------------------------------
    try:
        while True:
            # Advance the simulation by one timestep (typically 0.1s)
            sim_api.tick()

            # Get the next target waypoint from OpenCDA's local planner
            # (This is fed to the MPC as the reference trajectory)
            target_waypoint = ego_vehicle.routing_local_planner.waypoints_queue[0]

            # Run the MPC controller to compute throttle/steering/brake
            control_command = ego_vehicle.control_manager.run_step(
                target_speed=10.0,  # Target speed in m/s (e.g., 10 m/s = 36 km/h)
                waypoint=target_waypoint
            )

            # Apply the control command to the ego vehicle in CARLA
            ego_vehicle.vehicle.apply_control(control_command)

            # Optional: Add a small delay to control the simulation speed
            time.sleep(0.1)

    except KeyboardInterrupt:
        # Gracefully close the simulation on CTRL+C
        print("Shutting down...")
    finally:
        # Cleanup
        sim_api.close()

if __name__ == '__main__':
    run_scenario()