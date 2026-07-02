import queue
import time
from pathlib import Path

import carla
import cv2
import numpy as np


def carla_image_to_bgr(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3]


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.filter("model3")[0]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = spawn_points[10]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "960")
    camera_bp.set_attribute("image_size_y", "540")
    camera_bp.set_attribute("fov", "90")

    camera_transform = carla.Transform(
        carla.Location(x=1.5, z=2.4),
        carla.Rotation(pitch=-8)
    )

    camera = world.spawn_actor(
        camera_bp,
        camera_transform,
        attach_to=vehicle
    )

    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    vehicle.apply_control(
        carla.VehicleControl(
            throttle=0.25,
            steer=0.0,
            brake=0.0
        )
    )    

    Path("assets/images").mkdir(parents=True, exist_ok=True)
    saved = False

    print("Front camera demo running.")
    print("Press q in the OpenCV window to quit.")

    try:
        while True:
            image = image_queue.get(timeout=2.0)
            frame = carla_image_to_bgr(image)

            vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.25,
                    steer=0.0,
                    brake=0.0
                )
            )
            if not saved:
                cv2.imwrite("assets/images/front_camera_demo.png", frame)
                print("Saved assets/images/front_camera_demo.png")
                saved = True

            cv2.imshow("MiniFSD Front Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()
        print("Cleaned up actors.")


if __name__ == "__main__":
    main()
