import cv2
import numpy as np
import pyrealsense2 as rs
from perception import pixel_to_world, camera_to_world, init_realsense
from sklearn.cluster import DBSCAN

class ObstacleDetector:
    def __init__(self, pipeline, align, ground_height_limit=0.05):
        self.pipeline = pipeline
        self.align = align
        self.ground_height_limit = ground_height_limit

    def find_obstacle(self, visualize=False):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return [], None

        # pyrealsense filter
        spat_filter = rs.spatial_filter()
        temp_filter = rs.temporal_filter()
        hole_filling = rs.hole_filling_filter()
        depth_frame = spat_filter.process(depth_frame)
        depth_frame = temp_filter.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_scale = self.pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        depth_m = depth_image * depth_scale

        # detect edge
        depth_8u = cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        depth_8u = cv2.dilate(depth_8u, kernel, iterations=1)
        edges = cv2.Canny(depth_8u, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        obstacles = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 20 or h < 20:
                continue
            if x < 10 or y < 10 or x + w > depth_m.shape[1] - 10 or y + h > depth_m.shape[0] - 10:
                continue

            roi = depth_m[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            roi_mean = np.mean(roi[np.isfinite(roi)])

            expand = 10
            y1 = max(0, y - expand)
            y2 = min(depth_m.shape[0], y + h + expand)
            x1 = max(0, x - expand)
            x2 = min(depth_m.shape[1], x + w + expand)
            bg_roi = depth_m[y1:y2, x1:x2].copy()

            bg_roi[(expand):(expand+h), (expand):(expand+w)] = np.nan
            bg_mean = np.nanmean(bg_roi)

            if np.isnan(roi_mean) or np.isnan(bg_mean):
                continue
            
            # roi depth mean should be less than bg depth mean
            if roi_mean < bg_mean - 0.05:
                obstacles.append((x, y, w, h))
                if visualize:
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if visualize:
            return obstacles, color_image
        else:
            return obstacles
    
    def get_obstacle_3d(self, visualize=False):
        if visualize:
            obstacles, vis_img = self.find_obstacle(visualize=True)
        else:
            obstacles = self.find_obstacle(visualize=False)
        
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()

        if not depth_frame:
            return []

        boxes_3d = []

        for (x, y, w, h) in obstacles:
            world_points = []
            for i in range(y, y + h):
                for j in range(x, x + w):
                    depth = depth_frame.get_distance(j, i)
                    if depth == 0 or not np.isfinite(depth):
                        continue
                    camera_point = pixel_to_world(depth_frame, [j, i], depth)
                    world_point = camera_to_world(np.array(camera_point))
                    world_points.append(world_point)
            
            if not world_points:
                continue

            world_points = np.array(world_points)
            x_min, y_min, z_min = np.min(world_points, axis=0)
            x_max, y_max, z_max = np.max(world_points, axis=0)

            boxes_3d.append((x_min, y_min, z_min, x_max, y_max, z_max))

        if visualize:
            return boxes_3d, vis_img
        else:
            return boxes_3d


if __name__ == "__main__":
    pipeline, align = init_realsense()
    detector = ObstacleDetector(pipeline, align)
    vis_img = None

    try:
        while True:
            boxes_3d, vis_img = detector.get_obstacle_3d(visualize=True)
            for i, (x_min, y_min, z_min, x_max, y_max, z_max) in enumerate(boxes_3d):
                print(f"Box {i}: x({x_min:.3f}~{x_max:.3f}), y({y_min:.3f}~{y_max:.3f}), z({z_min:.3f}~{z_max:.3f})")
            if vis_img is not None:
                cv2.imshow("Obstacles", vis_img)
            if cv2.waitKey(1) == 27:
                break
    except KeyboardInterrupt:
        pipeline.stop()
        cv2.destroyAllWindows()
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
