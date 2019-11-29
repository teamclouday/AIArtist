# This script will increase the fps of the original video (fps 5)

import numpy as np
import cv2

video_path = "fps5_output.avi"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
fpsEnhance = 12
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'DIVX'), fps*fpsEnhance, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

frames = []

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frames.append(frame)
    else:
        break

cap.release()

assert total_frames == len(frames)
assert total_frames > 1

now = None

for i in range(1, total_frames):
    print("Converting frame ({0} / {1})".format(i+1, total_frames))
    prev = np.array(frames[i-1]).astype(np.int32)
    now = np.array(frames[i]).astype(np.int32)
    offset = (now - prev) / fpsEnhance
    writer.write(prev.astype(np.uint8))
    for j in range(1, fpsEnhance):
        newframe = (prev + j*offset).astype(np.int32)
        newframe = np.clip(newframe, 0, 255).astype(np.uint8)
        writer.write(newframe)

writer.write(now.astype(np.uint8))
writer.release()

print("Done")