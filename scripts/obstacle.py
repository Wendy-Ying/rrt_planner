import cv2
import numpy as np
import pyrealsense2 as rs
from perception import pixel_to_world, camera_to_world, init_realsense

class ObstacleDetector:
    def __init__(self, pipeline, align, ground_height_limit=0.05):
        self.pipeline = pipeline
        self.align = align
        self.ground_height_limit = ground_height_limit

    def get_obstacle(self, visualize=False):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return [], None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_scale = self.pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        depth_m = depth_image * depth_scale

        # find gradient
        depth_blur = cv2.GaussianBlur(depth_m, (5, 5), 0)
        sobelx = cv2.Sobel(depth_blur, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(depth_blur, cv2.CV_64F, 0, 1, ksize=5)
        grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        edge_mask = (grad_mag > 0.2).astype(np.uint8) * 255

        # generate region mask
        kernel = np.ones((5, 5), np.uint8)
        region_mask = cv2.dilate(edge_mask, kernel, iterations=2)
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes_3d = []
        h, w = depth_m.shape

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)

            border_margin = 30
            if x < border_margin or (x + bw) > (w - border_margin):
                continue

            roi_points = []
            for dy in range(bh):
                for dx in range(bw):
                    px = x + dx
                    py = y + dy
                    d = depth_m[py, px]
                    if d <= 0:
                        continue
                    try:
                        cam_pt = pixel_to_world(depth_frame, [px, py], d)
                        base_pt = camera_to_world(np.array(cam_pt))
                        if base_pt[2] < self.ground_height_limit:
                            continue
                        roi_points.append(base_pt)
                    except:
                        continue

            if len(roi_points) == 0:
                continue

            points_np = np.array(roi_points)
            x_min, y_min, z_min = np.min(points_np, axis=0)
            x_max, y_max, z_max = np.max(points_np, axis=0)
            boxes_3d.append([(x_min, y_min, z_min), (x_max, y_max, z_max)])

            if visualize:
                cv2.rectangle(color_image, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
                text = f"3D Box: {x_min:.2f},{y_min:.2f},{z_min:.2f}"
                cv2.putText(color_image, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        return boxes_3d if not visualize else boxes_3d, color_image


if __name__ == "__main__":
    pipeline, align = init_realsense()
    detector = ObstacleDetector(pipeline, align)

    try:
        while True:
            boxes_3d, vis_img = detector.get_obstacle(visualize=True)
            for i, (pt_min, pt_max) in enumerate(boxes_3d):
                print(f"Obstacle {i}: Min={pt_min}, Max={pt_max}")
            if vis_img is not None:
                cv2.imshow("Obstacles", vis_img)
            if cv2.waitKey(1) == 27:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
