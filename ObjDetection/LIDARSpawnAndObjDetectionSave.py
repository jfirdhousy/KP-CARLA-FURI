#!/usr/bin/env python3
"""
spawn_and_detect_with_csv.py

- Spawn ~100 vehicles (batch spawn)
- Spawn a hero vehicle (with LiDAR attached)
- Set spectator camera to focus on hero vehicle
- Listen to semantic LiDAR and:
  1) Save raw point cloud data to lidar_points.csv
  2) Save detection summary to object_log.csv
"""

import glob
import os
import sys
import time
import random
import math
import logging
import csv
from datetime import datetime

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import VehicleLightState as vls

# ----- Config ----- #
HOST = '127.0.0.1'
PORT = 2000
NUM_VEHICLES = 30
LIDAR_CHANNELS = 64
LIDAR_PPS = 56000
LIDAR_FREQ = 80
LIDAR_RANGE = 100.0
HERO_MODEL_FILTER = 'model3'
RAW_LIDAR_CSV = r"D:\KP_Carla\ObjDetection\lidar_points.csv"
DETECTION_LOG_CSV = r"D:\KP_Carla\ObjDetection\object_log.csv"
# ------------------ #

os.makedirs(os.path.dirname(RAW_LIDAR_CSV), exist_ok=True)

# semantic mapping
object_id = {"None": 0,
             "Buildings": 1,
             "Fences": 2,
             "Other": 3,
             "Pedestrians": 4,
             "Poles": 5,
             "RoadLines": 6,
             "Roads": 7,
             "Sidewalks": 8,
             "Vegetation": 9,
             "Vehicles": 10,
             "Wall": 11,
             "TrafficsSigns": 12,
             "Sky": 13,
             "Ground": 14,
             "Bridge": 15,
             "RailTrack": 16,
             "GuardRail": 17,
             "TrafficLight": 18,
             "Static": 19,
             "Dynamic": 20,
             "Water": 21,
             "Terrain": 22
             }
key_list = list(object_id.keys())
value_list = list(object_id.values())

actor_list = []
spawned_vehicle_ids = []

# buat header CSV (sekali saja)
with open(RAW_LIDAR_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "x", "y", "z", "object_tag"])

with open(DETECTION_LOG_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "detections_summary"])


def generate_lidar_blueprint(blueprint_library):
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast_semantic')
    lidar_bp.set_attribute('channels', str(LIDAR_CHANNELS))
    lidar_bp.set_attribute('points_per_second', str(LIDAR_PPS))
    lidar_bp.set_attribute('rotation_frequency', str(LIDAR_FREQ))
    lidar_bp.set_attribute('range', str(LIDAR_RANGE))
    return lidar_bp


def semantic_lidar_data(point_cloud_data):
    """Simpan data mentah + log deteksi objek ke CSV"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    # --- 1️⃣ Simpan raw point cloud ---
    raw_points = []
    for detection in point_cloud_data:
        try:
            tag = detection.object_tag
            x, y, z = detection.x, detection.y, detection.z
        except AttributeError:
            # fallback untuk versi CARLA lama
            tag = detection.object_tag
            x, y, z = detection.point.x, detection.point.y, detection.point.z
        raw_points.append([timestamp, x, y, z, tag])

    # tulis batch ke CSV
    with open(RAW_LIDAR_CSV, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(raw_points)

    # --- 2️⃣ Deteksi objek sederhana ---
    nearest = {}
    for detection in point_cloud_data:
        tag = detection.object_tag
        try:
            name = key_list[value_list.index(tag)]
        except ValueError:
            name = f"Unknown({tag})"
        try:
            x, y, z = detection.x, detection.y, detection.z
        except AttributeError:
            x, y, z = detection.point.x, detection.point.y, detection.point.z
        distance = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        if name not in nearest or distance < nearest[name]:
            nearest[name] = distance

    detections_summary = ""
    if nearest:
        items = sorted(nearest.items(), key=lambda x: x[1])[:8]
        detections_summary = ", ".join([f"{n}:{d:.2f}" for n, d in items])
        print(f"Detected (name : distance m) -> {detections_summary}")

    # --- 3️⃣ Simpan log deteksi ke CSV ---
    with open(DETECTION_LOG_CSV, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, detections_summary])


def set_spectator_to_vehicle(world, vehicle, distance_back=8.0, height=4.0):
    transform = vehicle.get_transform()
    forward = transform.get_forward_vector()
    spectator_loc = transform.location - forward * distance_back
    spectator_loc.z += height
    spectator_rot = carla.Rotation(pitch=-15.0, yaw=transform.rotation.yaw, roll=0.0)
    spectator_transform = carla.Transform(spectator_loc, spectator_rot)
    world.get_spectator().set_transform(spectator_transform)


def spawn_vehicles_batch(client, world, filter='vehicle.*', number=NUM_VEHICLES):
    blueprint_library = world.get_blueprint_library()
    blueprints = list(blueprint_library.filter(filter))  # <-- ubah ke list agar bisa di-shuffle
    if not blueprints:
        raise RuntimeError("No vehicle blueprints found for filter: " + filter)
    random.shuffle(blueprints)

    spawn_points = world.get_map().get_spawn_points()
    if len(spawn_points) == 0:
        raise RuntimeError("No spawn points in the map")

    n_spawn = min(number, len(spawn_points))
    random.shuffle(spawn_points)

    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    batch = []
    for n, transform in enumerate(spawn_points):
        if n >= n_spawn:
            break
        bp = random.choice(blueprints)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        if bp.has_attribute('driver_id'):
            driver = random.choice(bp.get_attribute('driver_id').recommended_values)
            bp.set_attribute('driver_id', driver)
        bp.set_attribute('role_name', 'autopilot')
        batch.append(SpawnActor(bp, transform).then(SetAutopilot(FutureActor, True, client.get_trafficmanager().get_port())))

    responses = client.apply_batch_sync(batch, True)
    spawned_ids = []
    for res in responses:
        if res.error:
            logging.warning("Spawn vehicle error: %s", res.error)
        else:
            spawned_ids.append(res.actor_id)
    return spawned_ids


def main():
    logging.basicConfig(level=logging.INFO)
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)

    try:
        world = client.get_world()
        traffic_manager = client.get_trafficmanager()
        traffic_manager.global_percentage_speed_difference(0.0)

        blueprint_library = world.get_blueprint_library()
        hero_bp_list = list(blueprint_library.filter(HERO_MODEL_FILTER))
        if len(hero_bp_list) == 0:
            hero_bp = random.choice(list(blueprint_library.filter('vehicle.*')))
            logging.info("Hero model not found; using random vehicle for hero.")
        else:
            hero_bp = hero_bp_list[0]

        spawn_point = world.get_map().get_spawn_points()[0]
        hero_vehicle = world.spawn_actor(hero_bp, spawn_point)
        actor_list.append(hero_vehicle)
        print(f"Spawned hero vehicle id={hero_vehicle.id}")

        hero_vehicle.set_autopilot(True, traffic_manager.get_port())

        print(f"Spawning {NUM_VEHICLES} other vehicles...")
        vehicle_ids = spawn_vehicles_batch(client, world, filter='vehicle.*', number=NUM_VEHICLES)
        spawned_vehicle_ids.extend(vehicle_ids)
        print(f"Spawned vehicles count: {len(vehicle_ids)}")

        lidar_bp = generate_lidar_blueprint(blueprint_library)
        lidar_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.2))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=hero_vehicle)
        actor_list.append(lidar)
        print("LiDAR attached to hero vehicle.")

        set_spectator_to_vehicle(world, hero_vehicle, distance_back=10.0, height=5.0)
        print("Spectator camera set to hero vehicle view.")

        # callback LiDAR
        lidar.listen(lambda data: semantic_lidar_data(data))

        print("Simulation running. Press Ctrl+C to stop.")
        settings = world.get_settings()
        synchronous = settings.synchronous_mode
        while True:
            if synchronous:
                world.tick()
            else:
                world.wait_for_tick()
            set_spectator_to_vehicle(world, hero_vehicle, distance_back=10.0, height=5.0)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Interrupted by user. Cleaning up...")
    finally:
        print("Destroying actors...")
        try:
            client.apply_batch([carla.command.DestroyActor(x) for x in spawned_vehicle_ids])
        except Exception:
            pass
        for actor in actor_list:
            try:
                actor.destroy()
            except Exception:
                pass
        print("Done cleanup.")


if __name__ == '__main__':
    main()
