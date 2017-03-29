"""Client module which uses the interfaces exposed by `tf_http_server`."""
import requests
import cv2
import imageio
import base64
import tf_http_server
import tensorflow as tf

# More info on OpenCV Codecs
# http://docs.opencv.org/3.1.0/dd/d43/tutorial_py_video_display.html

# @debug
RESTORE_PATH = '/home/mlrg/mcmaster-text-to-motion-database/tensorflow_models/human_pose_model/deployment_files/vgg_16_cascade'

FPS = 20.0
MAX_FRAMES = 5*FPS

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
    BATCH_SIZE = 1
    frame_feed = tf.placeholder(dtype=tf.uint8)
    logits, resized_image = tf_http_server._get_joint_position_inference_graph(
        frame_feed, BATCH_SIZE)
    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=RESTORE_PATH)
    assert latest_checkpoint is not None

    restorer = tf.train.Saver(var_list=tf.global_variables())
    restorer.restore(sess=session, save_path=latest_checkpoint)

    frame_count = 0
    while cap.isOpened() and (frame_count < MAX_FRAMES):
        ret, frame = cap.read()
        if ret == True:
            # @debug
            batch_heatmaps = tf_http_server._get_heatmaps_for_batch([frame],
                                                                    logits,
                                                                    resized_image,
                                                                    session,
                                                                    frame_feed,
                                                                    BATCH_SIZE)

            cv2.imshow('frame', batch_heatmaps[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # out.write(frame)
            writer.append_data(frame)
            frame_count += 1
        else:
            print('error at frame {}'.format(frame_count))
            break

    # out.release()
    writer.close()
    with open('output.avi', 'rb') as video_handle:
        video = video_handle.read()
        requests.post('http://localhost:8246/Video', base64.b64encode(video))
        # requests.get('http://localhost:8246/Video')

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
