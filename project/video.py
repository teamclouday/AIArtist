from StyleTransfer import *
import cv2

video_path = "videoplayback.webm"
style_path = "blue-and-red-abstract-painting-1799912.jpg"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'DIVX'), fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

count = 0

while(cap.isOpened()):
    if count == 5: break
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tf.keras.backend.clear_session()
        model = StyleTransfer(frame, style_path, video=True)
        opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
        model.transfer(opt)
        newframe = model.get_frame(model.img)
        frame = cv2.cvtColor(newframe, cv2.COLOR_RGB2BGR)
        cv2.imshow("frame", newframe)
        writer.write(newframe)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()