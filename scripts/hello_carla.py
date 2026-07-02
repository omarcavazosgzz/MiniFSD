import time
import math
import carla


def follow_vehicle_spectator(world, vehicle, distance=8.0, height=5.0):
    spectator = world.get_spectator()
    transform = vehicle.get_transform()

    yaw_rad = math.radians(transform.rotation.yaw)

    # Position camera behind vehicle using vehicle yaw
    offset_x = -distance * math.cos(yaw_rad)
    offset_y = -distance * math.sin(yaw_rad)

    camera_location = transform.location + carla.Location(
        x=offset_x,
        y=offset_y,
        z=height
    )

    camera_rotation = carla.Rotation(
        pitch=-25,
        yaw=transform.rotation.yaw,
        roll=0
    )

    spectator.set_transform(carla.Transform(camera_location, camera_rotation))


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.filter("model3")[0]
    spawn_point = world.get_map().get_spawn_points()[0]

    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True)

    print("Spawned Tesla Model 3.")
    print("Autopilot enabled.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            follow_vehicle_spectator(world, vehicle)
            time.sleep(0.03)

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        vehicle.destroy()
        print("Vehicle destroyed.")


if __name__ == "__main__":
    main()

