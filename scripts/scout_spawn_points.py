import argparse
from pathlib import Path

import carla
import cv2
import numpy as np

from src.simulator.carla_environment import CarlaEnvironment
from src.simulator.camera_manager import CameraManager


def make_contact_sheet(image_paths, output_path, thumb_width=320, thumb_height=180, cols=4):
    images = []

    for path in image_paths:
        img = cv2.imread(str(path))

        if img is None:
            continue

        img = cv2.resize(img, (thumb_width, thumb_height))
        images.append(img)

    if not images:
        raise RuntimeError("No images available for contact sheet.")

    rows = int(np.ceil(len(images) / cols))

    sheet = np.zeros(
        (rows * thumb_height, cols * thumb_width, 3),
        dtype=np.uint8,
    )

    for i, img in enumerate(images):
        row = i // cols
        col = i % cols

        y0 = row * thumb_height
        x0 = col * thumb_width

        sheet[y0:y0 + thumb_height, x0:x0 + thumb_width] = img

    cv2.imwrite(str(output_path), sheet)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First spawn index to test.",
    )

    parser.add_argument(
        "--end",
        type=int,
        default=60,
        help="Last spawn index to test, exclusive.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/images/spawn_scout",
        help="Folder for spawn point screenshots.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = CarlaEnvironment()
    spawn_points = env.map.get_spawn_points()

    print(f"Map: {env.map.name}")
    print(f"Total spawn points: {len(spawn_points)}")

    vehicle_bp = env.blueprint_library.filter("model3")[0]

    saved_images = []

    end = min(args.end, len(spawn_points))

    for index in range(args.start, end):
        print(f"Scouting spawn point {index}")

        vehicle = None
        camera_manager = None

        try:
            spawn_point = spawn_points[index]
            vehicle = env.world.spawn_actor(vehicle_bp, spawn_point)

            camera_manager = CameraManager(env, vehicle)
            camera_manager.attach_rgb_camera()

            # Apply a tiny brake so the vehicle stays still.
            vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    steer=0.0,
                    brake=1.0,
                )
            )

            # Let CARLA produce a few camera frames.
            frame = None
            for _ in range(5):
                frame = camera_manager.get_frame(timeout=5.0)

            frame = frame.copy()

            if frame is None:
                continue

            label = f"spawn {index}"
            cv2.putText(
                frame,
                label,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

            image_path = output_dir / f"spawn_{index:03d}.png"
            cv2.imwrite(str(image_path), frame)
            saved_images.append(image_path)

        except Exception as exc:
            print(f"Failed spawn {index}: {exc}")

        finally:
            if camera_manager is not None:
                camera_manager.destroy()

            if vehicle is not None:
                vehicle.destroy()

    contact_sheet_path = output_dir / f"spawn_contact_sheet_{args.start}_{end}.png"
    if saved_images:
        make_contact_sheet(saved_images, contact_sheet_path)
    else:
        print("No spawn screenshots were saved. No contact sheet created.")
        print()
        print(f"Saved screenshots to: {output_dir}")
        print(f"Saved contact sheet: {contact_sheet_path}")


if __name__ == "__main__":
    main()
