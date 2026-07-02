import carla


class CarlaEnvironment:
    def __init__(self, host="localhost", port=2000, timeout=10.0):
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)

        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()

    def get_spawn_point(self, index=20):
        spawn_points = self.map.get_spawn_points()

        if not spawn_points:
            raise RuntimeError("No spawn points found in CARLA map.")

        if index >= len(spawn_points):
            raise IndexError(
                f"Spawn index {index} out of range. "
                f"Map only has {len(spawn_points)} spawn points."
            )

        return spawn_points[index]

    def destroy_actors(self, actors):
        for actor in actors:
            if actor is not None:
                actor.destroy()
