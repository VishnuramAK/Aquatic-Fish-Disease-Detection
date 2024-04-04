# importing libraries
import streamlit as st
import streamlit_lottie as st_lottie
from streamlit_option_menu import option_menu
import cv2
import time
from sort import *
from ultralytics import YOLO
import numpy as np 
import matplotlib.pyplot as plt
import requests
import cvzone
from PIL import Image
import os

# set page config 
st.set_page_config(page_title='AquaticAI', page_icon=":fish:", layout='wide')

# loading animations
def loader_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# loading assets

front = loader_url('https://lottie.host/237cb9fd-cc62-4cc0-a5f4-98a8312ab345/RS0ikWn1jD.json')
image = loader_url('https://lottie.host/357cfbe4-8c2d-4691-9761-d9c8a1941e2a/imZTdeOzez.json')
video = loader_url('https://lottie.host/3f90dd6a-5250-4e46-a591-ec1d522f8c1e/LfrTqYpUcF.json')

# model
model = YOLO('best.pt')

# classnames
classnames = ['Columnaris Disease', 'EUS Disease', 'Gill Disease', 'Healthy Fish', 'streptococcus Disease']
#classnames = ['Disease', 'Good']

# tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [400, 297, 673, 297]
totalCount = []

# functions
def aquatic_image(image):
    img = Image.open(image)
    out = img.save('aqua.jpg')
    img_file = cv2.imread('aqua.jpg')
    result = model(img_file)[0]
    for r in result.boxes.data.tolist():
        x1,y1,x2,y2,score,class_id = r
        cv2.rectangle(img_file, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 2)
        st.image(img_file[:,:,::-1])
        word = classnames[int(class_id)]
    return word

def aquatic_video(file):
    cap = cv2.VideoCapture(file)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1800)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)

    while True:
        ret, frame = cap.read()
        detections = np.empty((0, 5))
        if not ret:
            break  # Break if no frame is read

        results = model(frame, show=False, save=True)[0]
        
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            
            cv2.rectangle(frame, (0,0), (200,50), (0,0,0),-1)
            
            cv2.putText(frame, classnames[int(class_id)], (int(x1)+20, int(y1)+20), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 5)
            
            if score > 0.7:
                #cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 5)
                cv2.putText(frame, f'{score:.2f}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                                    cv2.LINE_AA)
                currentArray = np.array([x1, y1, x2, y2, score])
                detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)
        
        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(35, y1)),
                            scale=2, thickness=3, offset=10)
                
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        
            if totalCount.count(id) == 0:
                totalCount.append(id)

        cv2.putText(frame,str(len(totalCount)),(100,50),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)
        cv2.imshow('Output', frame)
            
        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
# Home page sidebar
with st.sidebar:
    with st.container():
        i,j = st.columns((4,4))
        with i:
            st.empty()
        with j:
            st.empty()

    choose = option_menu(
        "AquaticAI",
        ['Home', 'Image', 'Video', 'Live'],
        menu_icon='vision',
        default_index=0,
        orientation='vertical'
    )

if choose == 'Home':
    
    st.markdown("<h1 style='text-align: center;'>AquaticAI</h1>", unsafe_allow_html=True)

    st.write('-------')

    st.markdown("""
            AquaticAI introduces FishNet, a YOLOv8-powered system revolutionizing disease detection in aquaculture. 
            By automating the identification of diseased fish, FishNet mitigates economic losses and enhances food security. 
            Traditional manual observation methods are replaced by real-time image processing, reducing human error and time consumption. 
            Through deep learning, FishNet contributes to sustainable fish farming practices, bolstering efficiency and safeguarding the aquaculture industry.
            """, unsafe_allow_html=True)
    st.lottie(front, height=400, key='yoga')
    

elif choose == 'Image':

    st.title('For Images')
    st.lottie(image, height=200, key='image')
    uploaded_file = st.file_uploader(label='Upload your image', type=['.jpg', '.jpeg', '.png'])
    
    if st.button('Convert') and uploaded_file is not None:
        out = aquatic_image(uploaded_file)
        st.markdown(f"<h2 style='text-align: center;'>{out}</h2>", unsafe_allow_html=True)   
        os.remove('aqua.jpg')

elif choose == 'Video':
    
    st.title("For Video")

    st.lottie(video, height=300, key='video')

    file = st.file_uploader(label='Upload your video', type=['.mp4', '.mkv', '.HEVC'])

    if st.button('Convert') and file is not None:
        aquatic_video(file.name)

elif choose == 'Live':
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1800)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)

    while True:
        
        ret, frame = cap.read()
        detections = np.empty((0, 5))
        results = model(frame, show=False)[0]
        
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            
            #cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255),4)
            cv2.rectangle(frame, (0,0), (200,50), (0,0,0),-1)
            
            cv2.putText(frame, classnames[int(class_id)], (int(x1)+20, int(y1)+20), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 5)
            
            if score > 0.7:
                #cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 5)
                cv2.putText(frame, f'{score:.2f}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                                    cv2.LINE_AA)
                currentArray = np.array([x1, y1, x2, y2, score])
                detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)
        
        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(35, y1)),
                            scale=2, thickness=3, offset=10)
                
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        
            if totalCount.count(id) == 0:
                totalCount.append(id)

        cv2.putText(frame,str(len(totalCount)),(100,50),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)
        cv2.imshow('Output', frame)
            
        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()