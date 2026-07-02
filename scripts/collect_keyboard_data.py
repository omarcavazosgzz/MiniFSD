import argparse

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


def draw_frame(screen, frame, speed, control, recording, frame_count, session_path):
    rgb = frame[:, :, ::-1]

    surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    screen.blit(surface, (0, 0))

    font = pygame.font.SysFont("Arial", 22)

    status = "REC" if recording else "PAUSED"

    lines = [
        f"Status: {status}",
        f"Frames saved: {frame_count}",
        f"Speed: {speed:.1f} km/h",
        f"Throttle: {control.throttle:.2f}",
        f"Brake: {control.brake:.2f}",
        f"Steer: {control.steer:.2f}",
        "W throttle | A/D steer | S brake",
        "R toggle recording | Q quit",
        f"Session: {session_path}",
    ]

    y = 10
    for line in lines:
        text = font.render(line, True, (255, 255, 255))
        screen.blit(text, (10, y))
        y += 26

    pygame.display.flip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session-name",
        type=str,
        default=None,
        help="Optional exact session folder name."
    )
    parser.add_argument(
        "--spawn-index",
        type=int,
        default=20,
        help="CARLA spawn point index."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    env = CarlaEnvironment()

    vehicle_manager = VehicleManager(env, spawn_index=args.spawn_index)
    vehicle = vehicle_manager.spawn()

    camera_manager = CameraManager(env, vehicle)
    camera_manager.attach_rgb_camera()

    controller = KeyboardController()
    recorder = DataRecorder(
        session_name=args.session_name,
        session_prefix="manual"
    )

    pygame.init()
    screen = pygame.display.set_mode((960, 540))
    pygame.display.set_caption("MiniFSD Keyboard Data Collection")
    clock = pygame.time.Clock()

    running = True
    recording = True

    print("Keyboard data collection running.")
    print("Click the pygame window first.")
    print("W throttle | A/D steer | S brake")
    print("R toggle recording | Q quit")
    print(f"Saving to: {recorder.base_dir}")

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False

                    if event.key == pygame.K_r:
                        recording = not recording
                        print("Recording:", recording)

            frame = camera_manager.get_frame()
            control = controller.get_control()

            vehicle.apply_control(control)

            speed = get_speed_kmh(vehicle)

            if recording:
                recorder.record(
                    image=frame,
                    steering=control.steer,
                    throttle=control.throttle,
                    brake=control.brake,
                    speed=speed,
                )

            draw_frame(
                screen=screen,
                frame=frame,
                speed=speed,
                control=control,
                recording=recording,
                frame_count=recorder.frame_id,
                session_path=recorder.base_dir,
            )

            clock.tick(20)

    finally:
        recorder.close()
        camera_manager.destroy()
        vehicle_manager.destroy()
        pygame.quit()
        print("Keyboard data collection finished.")
        print(f"Saved frames: {recorder.frame_id}")
        print(f"Session folder: {recorder.base_dir}")


if __name__ == "__main__":
    main()
