from perception import init_realsense, process_frame
import threading
import numpy as np
import time
from datetime import datetime

obstacle = None
lock = threading.Lock()
obstacle_updated_event = threading.Event()
stop_flag = threading.Event()

def renew_listener(pipeline, align, threshold=0.05):
    global obstacle
    prev_obstacle = None

    while not stop_flag.is_set():
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        new_obstacle = process_frame(color_image, depth_frame, mode='replan')

        if new_obstacle is not None:
            if prev_obstacle is None:
                with lock:
                    obstacle = new_obstacle
                    obstacle_updated_event.set()
                prev_obstacle = new_obstacle
                on_obstacle_changed(obstacle)
            else:
                diff = np.linalg.norm(np.array(new_obstacle) - np.array(prev_obstacle))
                if diff > threshold:
                    with lock:
                        obstacle = new_obstacle
                        obstacle_updated_event.set()
                    prev_obstacle = new_obstacle
                    on_obstacle_changed(obstacle)

        time.sleep(0.05)

def on_obstacle_changed(obstacle):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [Trigger] Responding immediately to obstacle change!")

def main():
    global obstacle

    pipeline, align = init_realsense()

    t = threading.Thread(target=renew_listener, args=(pipeline, align))
    t.daemon = True
    t.start()

    time.sleep(50)
    stop_flag.set()
    t.join()

if __name__ == "__main__":
    main()