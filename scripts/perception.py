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
    return pipeline, align, depth_scale

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
        [0, math.cos(0.52), -math.sin(0.52)],
        [0, math.sin(0.52), math.cos(0.52)]
    ])
    R2 = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    R = R1 @ R2
    t = np.array([0.4, -0.37, 0.47])
    return R @ camera_point + t

# --------------- Frame Processing ------------------

def process_frame(color_image, depth_frame, detector, depth_scale):
    """Detect ArUco markers and return obj/goal world coordinates."""
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=-30)

    corners, ids, _ = detector.detectMarkers(gray)
    obj, goal = None, None

    if ids is not None:
        for i, corner in enumerate(corners):
            points = corner[0]
            cx, cy = int(np.mean(points[:, 0])), int(np.mean(points[:, 1]))

            depth = depth_frame.get_distance(cx, cy)
            if depth == 0:
                continue

            camera_coords = pixel_to_world(depth_frame, [cx, cy], depth)
            world_coords = camera_to_world(np.array(camera_coords))

            marker_id = ids[i][0]
            if marker_id == 0:
                goal = world_coords
            elif marker_id == 2:
                obj = world_coords

            cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)

    return color_image, obj, goal

# --------------- Main Loop ------------------

def detect_loop():
    """Main detection loop."""
    pipeline, align, depth_scale = init_realsense()
    detector = set_aruco()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            color_image, obj, goal = process_frame(color_image, depth_frame, detector, depth_scale)

            cv2.imshow("Color Image", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 可选：在此处处理 obj 和 goal，例如：
            if obj is not None and goal is not None:
                print("Object:", obj)
                print("Goal:", goal)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

# --------------- Entry Point ------------------

if __name__ == "__main__":
    detect_loop()
