"""Client module which uses the interfaces exposed by `tf_http_server`."""
import time
import queue
import threading
import requests
import scipy.misc
import cv2
import numpy as np
import imageio
import base64
from human_pose_model.pose_utils import timethis
import tf_http_server
import tensorflow as tf

# More info on OpenCV Codecs
# http://docs.opencv.org/3.1.0/dd/d43/tutorial_py_video_display.html

# @debug
RESTORE_PATH = '/home/mlrg/mcmaster-text-to-motion-database/tensorflow_models/human_pose_model/deployment_files/vgg_16_cascade'

FPS = 10
BATCH_SIZE = 7
MAX_FRAMES = 2000*BATCH_SIZE
FRAME_QUEUE = queue.Queue()

@timethis.timethis
def _get_heatmap_batch_async(frames,
                             logits,
                             resized_image,
                             session,
                             frame_feed,
                             BATCH_SIZE):
    """
    """
    global FRAME_QUEUE

    frame_shape = frames[0].shape
    max_dim = max(frame_shape[0], frame_shape[1])
    min_dim = min(frame_shape[0], frame_shape[1])
    crop_width = int((max_dim - min_dim)/2)
    start = crop_width
    end = max_dim - crop_width
    cropped_frames = np.array(frames)[..., start:end, :]

    batch_heatmaps = tf_http_server._get_heatmaps_for_batch(cropped_frames,
                                                            logits,
                                                            resized_image,
                                                            session,
                                                            frame_feed,
                                                            BATCH_SIZE)

    FRAME_QUEUE = queue.Queue()
    for heatmap_index in range(len(batch_heatmaps)):
        FRAME_QUEUE.put((cropped_frames[heatmap_index], batch_heatmaps[heatmap_index]))


def run():
    """Runs camera test"""

    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'X264')
    # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))
    writer = imageio.get_writer('./output.avi',
                                'ffmpeg',
                                codec='h264',
                                fps=FPS)
    
    # @debug
    frame_feed = tf.placeholder(dtype=tf.uint8)
    logits, resized_image = tf_http_server._get_joint_position_inference_graph(
        frame_feed, BATCH_SIZE)
    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=RESTORE_PATH)
    assert latest_checkpoint is not None

    restorer = tf.train.Saver(var_list=tf.global_variables())
    restorer.restore(sess=session, save_path=latest_checkpoint)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    total_frame_count = 0
    frames = []
    while cap.isOpened() and (total_frame_count < MAX_FRAMES):
        ret, frame = cap.read()
        if ret == True:
            # @debug
            frames.append(frame)

            if len(frames) >= BATCH_SIZE:
                if total_frame_count >= 2*BATCH_SIZE:
                    batch_thread.join()

                args = (frames, logits, resized_image, session, frame_feed, BATCH_SIZE)
                batch_thread = threading.Thread(target=_get_heatmap_batch_async, args=args)
                batch_thread.start()
                for frame in frames:
                    FRAME_QUEUE.put(frame)

                frames = []

            time.sleep(0.65/BATCH_SIZE)
            if  not FRAME_QUEUE.empty():
                result = FRAME_QUEUE.get()
                next_frame = result[0]
                next_heatmap = result[1]

                # next_heatmap = cv2.resize(next_heatmap, next_frame.shape[0:2])
                next_heatmap = scipy.misc.imresize(next_heatmap,
                                                   next_frame.shape,
                                                   interp='bicubic')

                alpha = 0.5
                heatmap_on_image = cv2.addWeighted(next_frame,
                                                   alpha,
                                                   next_heatmap,
                                                   1.0 - alpha,
                                                   0.0)

                cv2.imshow('frame', heatmap_on_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


            # out.write(frame)
            # writer.append_data(frame)
            total_frame_count += 1
        else:
            print('error at frame {}'.format(total_frame_count))
            break

    # out.release()
    writer.close()
    # with open('output.avi', 'rb') as video_handle:
    #     video = video_handle.read()
    #     requests.post('http://localhost:8246/Video', base64.b64encode(video))
    #     # requests.get('http://localhost:8246/Video')

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
