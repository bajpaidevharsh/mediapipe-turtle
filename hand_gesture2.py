import cv2
import mediapipe
import turtle
import time
cap=cv2.VideoCapture(0)
hands=mediapipe.solutions.hands.Hands(False,2,1,0.5,0.5)
dev = turtle.Turtle()
screen = turtle.Screen()
screen.tracer(0)
while True:
    success,frame=cap.read()
    if success:
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame=cv2.flip(frame,1)
        results = hands.process(frame)
        if results.multi_hand_landmarks:
             for handlms in results.multi_hand_landmarks:
                mediapipe.solutions.drawing_utils.draw_landmarks(frame,handlms,mediapipe.solutions.hands.HAND_CONNECTIONS)
                index_finger_tip = handlms.landmark[8]
                h, w, _ = frame.shape
                cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                dev.goto(cx,cy*(-1))
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        cv2.imshow('output',frame)
        screen.update()
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
turtle.done()
