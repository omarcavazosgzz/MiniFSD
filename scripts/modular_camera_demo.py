import cv2

from src.simulator.carla_environment import CarlaEnvironment
from src.simulator.vehicle_manager import VehicleManager
from src.simulator.camera_manager import CameraManager


def main():
    env = CarlaEnvironment()

    vehicle_manager = VehicleManager(env, spawn_index=20)
    vehicle = vehicle_manager.spawn()

    camera_manager = CameraManager(env, vehicle)
    camera_manager.attach_rgb_camera()

    print("Modular camera demo running.")
    print("Press q to quit.")

    try:
        while True:
            frame = camera_manager.get_frame()
            vehicle_manager.apply_manual_control(
                throttle=0.25,
                steer=0.0,
                brake=0.0
            )

            cv2.imshow("MiniFSD Modular Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera_manager.destroy()
        vehicle_manager.destroy()
        cv2.destroyAllWindows()
        print("Cleaned up actors.")


if __name__ == "__main__":
    main()
