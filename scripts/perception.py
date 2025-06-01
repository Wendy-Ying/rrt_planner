import cv2
import pyrealsense2 as rs
import numpy as np
import math

# --------------- Configuration Modules ------------------

def init_realsense():
    """Initialize RealSense pipeline and streams."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    align = rs.align(rs.stream.color)
    return pipeline, align

def set_aruco():
    """Set up ArUco detector."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshConstant = 7
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.0
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    return detector

# --------------- Coordinate Conversion ------------------

def pixel_to_world(depth_frame, pixel, depth):
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    return rs.rs2_deproject_pixel_to_point(intrinsics, pixel, depth)

def camera_to_world(camera_point):
    R1 = np.array([
        [1, 0, 0],
        [0, math.cos(math.pi/6), -math.sin(math.pi/6)],
        [0, math.sin(math.pi/6), math.cos(math.pi/6)]
    ])
    R2 = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])
    R = R2 @ R1
    t = np.array([0.39, 0.337, 0.4])
    return R @ camera_point + t

# --------------- Frame Processing ------------------

def process_frame(color_image, depth_frame):
    """Detect ArUco markers and return obj/goal world coordinates."""
    detector = set_aruco()
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=-30)

    corners, ids, _ = detector.detectMarkers(gray)
    obj, goal, obstacle = None, None, None
    results = []

    if ids is not None:
        for i, corner in enumerate(corners):
            points = corner[0]
            cx, cy = int(np.mean(points[:, 0])), int(np.mean(points[:, 1]))

            depth = depth_frame.get_distance(cx, cy)
            if depth == 0:
                continue

            camera_coords = pixel_to_world(depth_frame, [cx, cy], depth)
            world_coords = camera_to_world(np.array(camera_coords))
            world_coords = np.array([world_coords[0], world_coords[1], world_coords[2]+0.01])
            
    #         marker_id = ids[i][0]
    #         if marker_id == 0:
    #             results.append((cx, cy, 'goal'))
    #         elif marker_id == 2:
    #             results.append((cx, cy, 'obj'))
    #         elif marker_id == 3:
    #             results.append((cx, cy, 'obstacle'))

    # return results  # List of (x, y, label)

            marker_id = ids[i][0]
            if marker_id == 0:
                goal = np.array([world_coords[0], world_coords[1], world_coords[2]+0.07])
            elif marker_id == 2:
                obj = world_coords
            elif marker_id == 3:
                obstacle = world_coords

    if obj is not None and goal is not None and obstacle is not None:
        return obj, goal, obstacle
    else:
        return None, None, None

def detect(pipeline, align):
    obj, goal = None, None
    while obj is None or goal is None:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        obj, goal, obstacle = process_frame(color_image, depth_frame)

        if obj is not None and goal is not None and obstacle is not None:
            return obj, goal, obstacle

if __name__ == "__main__":
    pipeline, align = init_realsense()
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            detected_points = process_frame(color_image, depth_frame)

            for (x, y, label) in detected_points:
                cv2.circle(color_image, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(color_image, label, (x + 10, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                camera_cords = pixel_to_world(depth_frame, [x, y], depth_frame.get_distance(x, y))
                world_coords = camera_to_world(np.array(camera_cords))
                print(f"{label} detected at: world={world_coords}")

            cv2.imshow("Color Image", color_image)
            if cv2.waitKey(1) == 27:  # ESC key
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
