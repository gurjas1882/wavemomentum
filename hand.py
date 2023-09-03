import cv2
import mediapipe as mp

import pyautogui

import uuid
import os
import time

import streamlit as st

st.set_page_config(page_title="Customize")
mp_hands = mp.solutions.hands

events = dict()

st.title("WaveMomentum")


for i in range(5):
    option = st.selectbox(
    'What gesture would you like to configure',
    ('Thumb_Up', 'Thumb_Down', 'Open_Palm', 'Closed_Fist', 'Pointing_Up'), key="a"+str(i))
    option2 = st.selectbox('What would you like this gesture to do',
    ('toggle-cursor', 'left', 'right', 'up', 'down', 'left-click', 'right-click', 'middle-click', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'), key="b"+str(i))

    events[i] = {
        "event": option,
        "action": option2,
    }
    st.text('-----------------------------')

detection_confidence_level = st.slider("detection confidence level",0.1,1.0,0.75)
st.write("detection confidence level is ",detection_confidence_level)

tracking_confidence_level = st.slider("tracking confidence level",0.1,1.0,0.5)
st.write("tracking confidence level is ",tracking_confidence_level)

cooldown = st.slider("cooldown time",0.1,10.0,0.5)
st.write("cooldown time is ",cooldown)


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

frameTimeStamp = 0

cap = cv2.VideoCapture(0)

lastTime = 0
lastTimeBack = 0

toggle = False

stframe = st.empty()

def result(result, image, ms):
    global lastTime
    global lastTimeBack
    global toggle
    for gestures in result.gestures:
        for event in events:
            print("event", events[event]["event"], "action", events[event]["action"], gestures[0].category_name)
            if gestures[0].category_name == events[event]["event"] and time.time() - lastTime >= cooldown:
                if "click" in events[event]["action"]:
                    pyautogui.click(button=str(events[event]["action"]).replace("-click", ""))
                elif events[event]["action"] == "toggle-cursor":
                    toggle = not toggle
                else:
                    pyautogui.press(events[event]["action"])
                    print("up")
                lastTime = time.time()




with mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5, max_num_hands = 1) as hands:

    with mp.tasks.vision.GestureRecognizer.create_from_options(
        mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="gestures.task"),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            min_hand_detection_confidence=detection_confidence_level,
            min_tracking_confidence=tracking_confidence_level,
            result_callback=result
        )
    ) as recog:
        while cap.isOpened():
            ret,frame = cap.read()
            image = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            multi = 1
            threshold = 0.015


            if results.multi_hand_landmarks and toggle:
                pyautogui.moveTo(results.multi_hand_landmarks[0].landmark[8].x * pyautogui.size().width * multi, results.multi_hand_landmarks[0].landmark[8].y * pyautogui.size().height * multi)


            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = image)




            frameTimeStamp += 1

            recog.recognize_async(mp_image, frameTimeStamp)

            stframe.image(image, channels="BGR", use_column_width=True)
            

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break


cap.release()
cv2.destroyAllWindows
