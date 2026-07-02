import cv2
import carla

from src.simulator.carla_environment import CarlaEnvironment
from src.simulator.vehicle_manager import VehicleManager
from src.simulator.camera_manager import CameraManager
from src.dataset.recorder import DataRecorder


def get_speed_kmh(vehicle):
    velocity = vehicle.get_velocity()
    speed_mps = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5
    return speed_mps * 3.6


def main():
    env = CarlaEnvironment()

    vehicle_manager = VehicleManager(env, spawn_index=20)
    vehicle = vehicle_manager.spawn()

    camera_manager = CameraManager(env, vehicle)
    camera_manager.attach_rgb_camera()

    recorder = DataRecorder(session_name="session_001")

    throttle = 0.25
    steer = 0.0
    brake = 0.0

    print("Collecting data.")
    print("Press q to stop.")

    try:
        while True:
            frame = camera_manager.get_frame()

            control = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake
            )

            vehicle.apply_control(control)

            speed = get_speed_kmh(vehicle)

            recorder.record(
                image=frame,
                steering=steer,
                throttle=throttle,
                brake=brake,
                speed=speed
            )

            cv2.imshow("MiniFSD Data Collection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        recorder.close()
        camera_manager.destroy()
        vehicle_manager.destroy()
        cv2.destroyAllWindows()
        print("Data collection finished.")


if __name__ == "__main__":
    main()

