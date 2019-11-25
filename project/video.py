from StyleTransfer import *
import cv2

video_path = "videoplayback.webm"
style_path = "blue-and-red-abstract-painting-1799912.jpg"
cap = cv2.VideoCapture(video_path)
_, frame = cap.read()
size = frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)
writer = cv2.VideoWriter("output.avi", -1, fps, size)

while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None: break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tf.keras.backend.clear_session()
    model = StyleTransfer(frame, style_path, video=True)
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    model.transfer(opt)
    newframe = model.get_frame(model.img)
    writer.write(newframe)

cap.release()
writer.release()