import pygame
import carla


class KeyboardController:
    def __init__(self):
        self.steer = 0.0
        self.steer_increment = 0.04
        self.max_steer = 0.7

    def get_control(self):
        keys = pygame.key.get_pressed()

        throttle = 0.0
        brake = 0.0

        if keys[pygame.K_w]:
            throttle = 0.35

        if keys[pygame.K_s]:
            brake = 0.7

        if keys[pygame.K_a]:
            self.steer -= self.steer_increment
        elif keys[pygame.K_d]:
            self.steer += self.steer_increment
        else:
            self.steer *= 0.75

        self.steer = max(-self.max_steer, min(self.max_steer, self.steer))

        if abs(self.steer) < 0.01:
            self.steer = 0.0

        return carla.VehicleControl(
            throttle=throttle,
            steer=self.steer,
            brake=brake
        )
