import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from src.simulator.carla_environment import CarlaEnvironment


def world_to_image(location, min_x, max_y, scale, padding):
    x = int((location.x - min_x) * scale + padding)
    y = int((max_y - location.y) * scale + padding)
    return x, y


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--waypoint-distance", type=float, default=2.0)
    parser.add_argument("--scale", type=float, default=6.0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/images/spawn_map",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = CarlaEnvironment()
    carla_map = env.map

    spawn_points = carla_map.get_spawn_points()
    waypoints = carla_map.generate_waypoints(args.waypoint_distance)

    if args.end is None:
        args.end = len(spawn_points)

    end = min(args.end, len(spawn_points))

    selected_spawn_points = list(enumerate(spawn_points))[args.start:end]

    all_locations = [wp.transform.location for wp in waypoints]
    all_locations += [sp.location for _, sp in selected_spawn_points]
    
    min_x = min(loc.x for loc in all_locations)
    max_x = max(loc.x for loc in all_locations)
    min_y = min(loc.y for loc in all_locations)
    max_y = max(loc.y for loc in all_locations)

    padding = 80

    width = int((max_x - min_x) * args.scale + 2 * padding)
    height = int((max_y - min_y) * args.scale + 2 * padding)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (245, 245, 245)

    # Draw lane waypoint cloud.
    for wp in waypoints:
        loc = wp.transform.location
        x, y = world_to_image(loc, min_x, max_y, args.scale, padding)
        cv2.circle(canvas, (x, y), 1, (180, 180, 180), -1)

    # Draw selected spawn points.
    csv_path = output_dir / f"spawn_points_{args.start}_{end}.csv"

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["index", "x", "y", "z", "yaw"])

        for index, spawn_point in selected_spawn_points:
            loc = spawn_point.location
            rot = spawn_point.rotation

            x, y = world_to_image(loc, min_x, max_y, args.scale, padding)

            cv2.circle(canvas, (x, y), 7, (0, 0, 255), -1)
            cv2.circle(canvas, (x, y), 10, (255, 255, 255), 2)

            yaw_rad = np.deg2rad(rot.yaw)
            arrow_len = 22

            end_x = int(x + arrow_len * np.cos(yaw_rad))
            end_y = int(y - arrow_len * np.sin(yaw_rad))

            cv2.arrowedLine(
                canvas,
                (x, y),
                (end_x, end_y),
                (255, 0, 0),
                2,
                tipLength=0.35,
            )

            cv2.putText(
                canvas,
                str(index),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

            writer.writerow([
                index,
                f"{loc.x:.3f}",
                f"{loc.y:.3f}",
                f"{loc.z:.3f}",
                f"{rot.yaw:.3f}",
            ])

    title = f"{carla_map.name} spawn points {args.start}-{end - 1}"
    cv2.putText(
        canvas,
        title,
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    image_path = output_dir / f"spawn_points_map_{args.start}_{end}.png"
    cv2.imwrite(str(image_path), canvas)

    print(f"Map: {carla_map.name}")
    print(f"Total spawn points: {len(spawn_points)}")
    print(f"Saved map: {image_path}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
