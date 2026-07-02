import carla


class VehicleManager:
    def __init__(self, env, vehicle_filter="model3", spawn_index=20):
        self.env = env
        self.vehicle_filter = vehicle_filter
        self.spawn_index = spawn_index
        self.vehicle = None

    def spawn(self):
        vehicle_bp = self.env.blueprint_library.filter(self.vehicle_filter)[0]
        spawn_point = self.env.get_spawn_point(self.spawn_index)

        self.vehicle = self.env.world.spawn_actor(vehicle_bp, spawn_point)
        return self.vehicle

    def apply_manual_control(self, throttle=0.25, steer=0.0, brake=0.0):
        if self.vehicle is None:
            raise RuntimeError("Vehicle has not been spawned.")

        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake
        )
        self.vehicle.apply_control(control)

    def destroy(self):
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None

