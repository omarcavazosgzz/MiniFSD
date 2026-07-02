import queue

import carla
import numpy as np


class CameraManager:
    def __init__(
        self,
        env,
        vehicle,
        image_width=960,
        image_height=540,
        fov=90,
        location_x=1.5,
        location_z=2.4,
        pitch=-8,
    ):
        self.env = env
        self.vehicle = vehicle
        self.image_width = image_width
        self.image_height = image_height
        self.fov = fov
        self.location_x = location_x
        self.location_z = location_z
        self.pitch = pitch

        self.camera = None
        self.image_queue = queue.Queue()

    def attach_rgb_camera(self):
        camera_bp = self.env.blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.image_width))
        camera_bp.set_attribute("image_size_y", str(self.image_height))
        camera_bp.set_attribute("fov", str(self.fov))

        camera_transform = carla.Transform(
            carla.Location(x=self.location_x, z=self.location_z),
            carla.Rotation(pitch=self.pitch)
        )

        self.camera = self.env.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle
        )

        self.camera.listen(self.image_queue.put)
        return self.camera

    def get_frame(self, timeout=2.0):
        image = self.image_queue.get(timeout=timeout)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        return array[:, :, :3]

    def destroy(self):
        if self.camera is not None:
            self.camera.stop()
            self.camera.destroy()
            self.camera = None
