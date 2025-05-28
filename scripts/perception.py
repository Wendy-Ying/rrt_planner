import cv2
import pyrealsense2 as rs
import numpy as np
import math


def init_realsense():
    """Initialize RealSense pipeline and streams."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # Get depth scale
    depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Create align object to align depth frame to color frame
    align_to = rs.stream.color
    align = rs.align(align_to)

    return pipeline, align, depth_scale

def set_aruco():
    """Set up ArUco detection parameters."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_params.adaptiveThreshConstant = 7
    aruco_params.minMarkerPerimeterRate = 0.02
    aruco_params.maxMarkerPerimeterRate = 4.0
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    return aruco_detector

def pixel_to_world(depth_frame, pixel, depth):
    """Convert pixel coordinates to 3D world coordinates."""
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    x, y, z = rs.rs2_deproject_pixel_to_point(intrinsics, pixel, depth)
    return x, y, z

def camera_to_world(camera_point,):
    """Convert camera coordinates to world coordinates."""
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
    world_point = R @ camera_point + t
    return world_point

def detect():
    """Main detection loop."""
    pipeline, align, depth_scale = init_realsense()

    # Prepare ArUco detector
    aruco_detector = set_aruco()

    try:
        while True:
            # get color and depth frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # Convert to grayscale and enhance contrast
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.convertScaleAbs(gray_image, alpha=1.2, beta=-30)

            # Detect ArUco markers
            corners, ids, _ = aruco_detector.detectMarkers(gray_image)

            if ids is not None:
                for i, corner in enumerate(corners):
                    points = corner[0]
                    cx = int(np.mean(points[:, 0]))
                    cy = int(np.mean(points[:, 1]))

                    depth = depth_frame.get_distance(cx, cy)
                    if depth == 0:
                        continue

                    x, y, z = pixel_to_world(depth_frame, [cx, cy], depth)
                    world_point = camera_to_world(np.array([x, y, z]))

                    print(f"ArUco ID {ids[i][0]} position (world): "
                          f"x={world_point[0]:.3f} m, y={world_point[1]:.3f} m, z={world_point[2]:.3f} m")

                    cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)

            cv2.imshow('Color Image', color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detect()
