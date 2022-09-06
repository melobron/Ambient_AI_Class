import argparse
import cv2
import os

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    args = parser.parse_args()

    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    inference_size = input_size(interpreter)

    cap = cv2.VideoCapture('../../video_for_coral_practice.mp4')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter('largest_face.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))

    print("Converting video...", end='')
    while cap.isOpened():
        print(".", end='')
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, args.threshold)[:args.top_k]
        cv2_im = detect_faces(cv2_im, inference_size, objs)

        out.write(cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("\nFinished!")
    cap.release()
    cv2.destroyAllWindows()

def detect_faces(cv2_im, inference_size, objs):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    total_area = height*width
    largest_face = {'score':0, 'bbox': ((0,0),(0,0))}
    largest_idx = -1

    # Find the largest face
    for idx, obj in enumerate(objs):
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        l_x0, l_y0 = largest_face['bbox'][0]
        l_x1, l_y1 = largest_face['bbox'][1]
        
        largest_size = (l_x1-l_x0)*(l_y1-l_y0)
        cur_size = (x1-x0) * (y1-y0)

        if cur_size > largest_size:
            largest_face['bbox'] = (x0,y0), (x1,y1)
            largest_face['score'] = int(100 * obj.score)
            largest_idx = idx
    # Draw bounding boxes
    for idx, obj in enumerate(objs):
        if idx == largest_idx: continue
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0,y0 = int(bbox.xmin), int(bbox.ymin)
        x1,y1 = int(bbox.xmax), int(bbox.ymax)
        cv2_im = cv2.rectangle(cv2_im, (x0,y0), (x1,y1), (0,255,0), -1)

    cv2_im = cv2.rectangle(cv2_im, largest_face['bbox'][0], largest_face['bbox'][1], (0,0,255),2)


    return cv2_im

if __name__ == '__main__':
    main()
