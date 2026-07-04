import argparse
from pathlib import Path

import carla
import cv2
import numpy as np
import pygame

from src.simulator.carla_environment import CarlaEnvironment
from src.simulator.vehicle_manager import VehicleManager
from src.simulator.camera_manager import CameraManager
from src.inference.steering_agent import SteeringAgent


def get_speed_kmh(vehicle):
    velocity = vehicle.get_velocity()
    speed_mps = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5
    return speed_mps * 3.6


def shape_steering(
    raw_steer,
    low_gain=1.0,
    high_gain=3.0,
    deadzone=0.025,
    curve_threshold=0.08,
    max_steer=0.55,
):
    abs_steer = abs(raw_steer)

    if abs_steer < deadzone:
        return 0.0

    if abs_steer >= curve_threshold:
        gain = high_gain
    else:
        blend = (abs_steer - deadzone) / (curve_threshold - deadzone)
        gain = low_gain + blend * (high_gain - low_gain)

    shaped = raw_steer * gain
    return float(np.clip(shaped, -max_steer, max_steer))


def draw_frame(screen, frame, speed, raw_steer, shaped_steer, applied_steer, throttle, brake, recording):
    rgb = frame[:, :, ::-1]
    surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    screen.blit(surface, (0, 0))

    font = pygame.font.SysFont("Arial", 22)

    lines = [
        "MiniFSD Neural Steering Test",
        f"Speed: {speed:.1f} km/h",
        f"Raw steering: {raw_steer:.3f}",
        f"Shaped steering: {shaped_steer:.3f}",
        f"Applied steering: {applied_steer:.3f}",
        f"Throttle: {throttle:.2f}",
        f"Brake: {brake:.2f}",
        f"Recording: {'ON' if recording else 'OFF'}",
        "Q quit",
    ]

    y = 10
    for line in lines:
        text = font.render(line, True, (255, 255, 255))
        screen.blit(text, (10, y))
        y += 28

    pygame.display.flip()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/steering_cnn_best.pt",
    )

    parser.add_argument(
        "--spawn-index",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--target-speed",
        type=float,
        default=18.0,
    )

    parser.add_argument(
        "--throttle",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--low-steering-gain",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--high-steering-gain",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--steering-deadzone",
        type=float,
        default=0.025,
    )

    parser.add_argument(
        "--curve-threshold",
        type=float,
        default=0.08,
    )

    parser.add_argument(
        "--max-steer",
        type=float,
        default=0.55,
    )

    parser.add_argument(
        "--straight-decay",
        type=float,
        default=0.55,
    )

    parser.add_argument(
        "--record",
        action="store_true",
        help="Record demo video to assets/videos.",
    )

    parser.add_argument(
        "--video-name",
        type=str,
        default="base_model_outside_loop.mp4",
        help="Output video filename when --record is enabled.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    env = CarlaEnvironment()

    vehicle_manager = VehicleManager(env, spawn_index=args.spawn_index)
    vehicle = vehicle_manager.spawn()

    camera_manager = CameraManager(env, vehicle)
    camera_manager.attach_rgb_camera()

    agent = SteeringAgent(checkpoint_path=args.checkpoint)

    pygame.init()
    screen = pygame.display.set_mode((960, 540))
    pygame.display.set_caption("MiniFSD Model Driving")
    clock = pygame.time.Clock()

    video_writer = None
    video_path = None

    if args.record:
        video_dir = Path("assets/videos")
        video_dir.mkdir(parents=True, exist_ok=True)

        video_path = video_dir / args.video_name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            20.0,
            (960, 540),
        )

        print(f"Recording demo to: {video_path}")

    running = True
    smoothed_steer = 0.0

    print("Model driving started.")
    print("Click the pygame window.")
    print("Press Q to quit.")

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False

            frame = camera_manager.get_frame()
            raw_steer = agent.predict(frame)

            shaped_steer = shape_steering(
                raw_steer=raw_steer,
                low_gain=args.low_steering_gain,
                high_gain=args.high_steering_gain,
                deadzone=args.steering_deadzone,
                curve_threshold=args.curve_threshold,
                max_steer=args.max_steer,
            )

            if shaped_steer == 0.0:
                smoothed_steer *= args.straight_decay
            else:
                smoothed_steer = (
                    args.smoothing * smoothed_steer
                    + (1.0 - args.smoothing) * shaped_steer
                )

            speed = get_speed_kmh(vehicle)

            if speed > args.target_speed:
                throttle = 0.0
                brake = 0.15
            else:
                throttle = args.throttle
                brake = 0.0

            control = carla.VehicleControl(
                throttle=throttle,
                steer=smoothed_steer,
                brake=brake,
            )

            vehicle.apply_control(control)

            draw_frame(
                screen=screen,
                frame=frame,
                speed=speed,
                raw_steer=raw_steer,
                shaped_steer=shaped_steer,
                applied_steer=smoothed_steer,
                throttle=throttle,
                brake=brake,
                recording=args.record,
            )

            if video_writer is not None:
                pygame_frame = pygame.surfarray.array3d(screen)
                pygame_frame = np.transpose(pygame_frame, (1, 0, 2))
                pygame_frame = cv2.cvtColor(pygame_frame, cv2.COLOR_RGB2BGR)
                video_writer.write(pygame_frame)

            clock.tick(20)

    finally:
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

        if video_writer is not None:
            video_writer.release()
            print(f"Saved video: {video_path}")

        camera_manager.destroy()
        vehicle_manager.destroy()
        pygame.quit()
        print("Model driving finished.")


if __name__ == "__main__":
    main()