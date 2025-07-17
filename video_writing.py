import cv2
from ultralytics import YOLO
from pathlib import Path
import numpy as np

path = Path('')

seg_model = YOLO('yolov8n.pt')



def pre_processar(frame, x1_ant, y1_ant, x2_ant, y2_ant):
  result = seg_model.predict(frame, classes=[0], stream=True, verbose=False)
  for r in result:
    boxes = r.boxes
    if(len(boxes) != 0):
        x1, y1, x2, y2 = boxes[0].xyxy[0].tolist()
 
        if (x1 - 50) > 0:
            x1 -= 50
        if (x2 + 50) < 1280:
            x2 += 50
        if (y1 - 50) > 0:
            y1 -= 50
        if (y2 + 50) < 720:
            y2 += 50

        #print('entrei aqui')
        frame = frame[int(y1):int(y2),int(x1):int(x2)]
    else:
        #print('nao foi detectado, pegando frames anteriores')
        x1, y1, x2, y2 = x1_ant, y1_ant, x2_ant, y2_ant

        if (x1 - 10) > 0:
            x1 -= 10
        if (x2 + 10) < 1280:
            x2 += 10
        if (y1 - 10) > 0:
            y1 -= 10
        if (y2 + 10) < 720:
            y2 += 10

        
        frame = frame[int(y1):int(y2),int(x1):int(x2)]

        

  return frame, x1, y1, x2, y2 

video_paths = list(path.glob('*.avi'))
print(video_paths)

for path in video_paths:

    output_path = 'c_' + str(path)
    print(output_path)

    cap = cv2.VideoCapture(str(path))

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))

    print(frame_width)
    print(frame_height)
    print(fps)
    print(fourcc_int)

    out = cv2.VideoWriter(output_path, fourcc_int, fps, (frame_width, frame_height))
    
    x1_ant, y1_ant, x2_ant, y2_ant = 0, 0, 0, 0
   
    #flag vira true quando detectar o primeiro frame
    ja_detectou = False

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame, x1_ant, y1_ant, x2_ant, y2_ant = pre_processar(frame, x1_ant, y1_ant, x2_ant, y2_ant)
            if frame.shape[0] != frame_height or frame.shape[1] != frame_width:
                frame = cv2.resize(frame,(frame_width, frame_height), interpolation=cv2.INTER_AREA) # INTER_AREA is good for shrinking
                #print('shape diferente')
                ja_detectou = True
            if ja_detectou:
                out.write(frame)
            frame_ant = frame
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("\nVideo processing complete!")
    print(f"Output video saved to: {output_path}")


