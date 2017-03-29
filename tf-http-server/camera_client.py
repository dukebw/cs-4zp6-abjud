"""Client module which uses the interfaces exposed by `tf_http_server`."""
import requests
import cv2
import base64

# More info on OpenCV Codecs
# http://docs.opencv.org/3.1.0/dd/d43/tutorial_py_video_display.html

MAX_FRAMES = 200

def run():
    """Runs camera test"""

    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
    
    frame_count = 0

    while cap.isOpened() and (frame_count < MAX_FRAMES):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('frame', frame)
            out.write(frame)
            frame_count += 1            
        else:
            break

    out.release()
    with open('output.avi', 'rb') as video_handle:
        video = video_handle.read()
        # requests.post('http://localhost:8246/Video', base64.b64encode(video))
        requests.get('http://localhost:8246/Video')

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
