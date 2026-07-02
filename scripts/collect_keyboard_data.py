import pygame
import numpy as np

from src.simulator.carla_environment import CarlaEnvironment
from src.simulator.vehicle_manager import VehicleManager
from src.simulator.camera_manager import CameraManager
from src.dataset.recorder import DataRecorder
from src.control.keyboard_controller import KeyboardController


def get_speed_kmh(vehicle):
    velocity = vehicle.get_velocity()
    speed_mps = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5
    return speed_mps * 3.6


def draw_frame(screen, frame, speed, control, recording=True):
    # CARLA frame is BGR. Pygame expects RGB.
    rgb = frame[:, :, ::-1]

    surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    screen.blit(surface, (0, 0))

    font = pygame.font.SysFont("Arial", 22)

    lines = [
        f"Speed: {speed:.1f} km/h",
        f"Throttle: {control.throttle:.2f}",
        f"Brake: {control.brake:.2f}",
        f"Steer: {control.steer:.2f}",
        "W throttle | A/D steer | S brake | Q quit",
        "REC" if recording else "NOT RECORDING",
    ]

    y = 10
    for line in lines:
        text = font.render(line, True, (255, 255, 255))
        screen.blit(text, (10, y))
        y += 26

    pygame.display.flip()


def main():
    env = CarlaEnvironment()

    vehicle_manager = VehicleManager(env, spawn_index=20)
    vehicle = vehicle_manager.spawn()

    camera_manager = CameraManager(env, vehicle)
    camera_manager.attach_rgb_camera()

    controller = KeyboardController()
    recorder = DataRecorder(session_name="manual_session_001")

    pygame.init()
    screen = pygame.display.set_mode((960, 540))
    pygame.display.set_caption("MiniFSD Keyboard Data Collection")
    clock = pygame.time.Clock()

    running = True

    print("Keyboard data collection running.")
    print("Click the pygame window first.")
    print("W throttle | A/D steer | S brake | Q quit")

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False

            frame = camera_manager.get_frame()
            control = controller.get_control()

            vehicle.apply_control(control)

            speed = get_speed_kmh(vehicle)

            recorder.record(
                image=frame,
                steering=control.steer,
                throttle=control.throttle,
                brake=control.brake,
                speed=speed,
            )

            draw_frame(screen, frame, speed, control)

            clock.tick(20)

    finally:
        recorder.close()
        camera_manager.destroy()
        vehicle_manager.destroy()
        pygame.quit()
        print("Keyboard data collection finished.")


if __name__ == "__main__":
    main()
