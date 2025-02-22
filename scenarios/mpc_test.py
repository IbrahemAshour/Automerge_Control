def run_scenario():
    # Initialize OpenCDA
    sim_api = SimAPI(config)
    ego_vehicle = sim_api.spawn_vehicle()

    # Set destination 200m ahead
    destination = carla.Location(x=200, y=0, z=0)
    sim_api.set_destination(ego_vehicle, destination)

    # Main loop
    try:
        while True:
            sim_api.tick()  # Advance simulation
            # Get control from MPC
            control = ego_vehicle.control_manager.run_step(
                target_speed=10.0,
                waypoint=ego_vehicle.routing_local_planner.waypoints_queue[0]
            )
            # Apply control to CARLA vehicle
            ego_vehicle.vehicle.apply_control(control)
    finally:
        sim_api.close()