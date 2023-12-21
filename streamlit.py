# Import all of the dependencies
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
from PIL import Image
from FaceRecogKnn import predict
import time



# Set the layout to the streamlit app as wide
st.set_page_config(layout='wide')
# Create a title for your app
col1, col2 = st.columns([2,1])

# В левом столбце размещаем заголовок и описание
col1.title("FaceApp")
col1.markdown("### **The app that recognizes faces**")

# В правом столбце размещаем изображение
col2.image(Image.open('facial-recognition.jpg'), width=300)

# Setup the sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('FaceApp')
    st.info('This application is originally developed from the Face recognition deep learning model.')
    st.sidebar.subheader('Parameters')
    # creating a button for webcam
    use_webcam = st.sidebar.button('Use Webcam')
    # creating a slider for detection confidence
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)

    # file uploader
    picture = st.sidebar.file_uploader("Upload a picture", type=["jpg", "png"])


st.info('The video below displays webcam feed')


# webrtc_streamer (key='sample')
class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        start_frame = time.time()
        # Flip the image (optional)
        frame = cv2.flip(img, 1)  # 0 = horizontal ,1 = vertical , -1 = both
        frame_copy = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        predictions = predict(frame_copy, model_path="classifier/trained_knn.clf")  # add path here
        font = cv2.FONT_HERSHEY_DUPLEX

        for name, (top, right, bottom, left), accuracy in predictions:
            top *= 4  # scale back the frame since it was scaled to 1/4 in size
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 255), 2)
            name = name + " " + str(round(accuracy, 2)) + "%"
            size = (right - left) / 220
            cv2.putText(img, name, (left - 10, top - 6), font, size, (255, 255, 255), 1)
            print(name)

            end_frame = time.time()
            fps = 1 / (end_frame - start_frame)
            cv2.rectangle(img, (15, 20), (125, 45), (0, 255, 255), -1)
            fps = "FPS: " + str(round(fps, 2))
            cv2.putText(img, fps, (16, 40), font, 0.5, (0, 0, 0), 1)

            if name == "unknown":
                cv2.putText(img,"ACCESS DENIED",(250, 400), font, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(img, "ACCESS PERMITTED",(250, 400), font, 0.8, (0, 255, 0), 2)


            return av.VideoFrame.from_ndarray(img, format='bgr24')


webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
               rtc_configuration=RTCConfiguration(
                   {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]} ))

