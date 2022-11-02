from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading
from sympy import content
import mediapipe as mp


def home(request):
    return render(request, 'index.html')


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            thickness=1, circle_radius=1)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        with self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            while True:
                (self.grabbed, self.frame) = self.video.read()
                self.frame = cv2.flip(self.frame, 1)
                self.results = face_mesh.process(self.frame)

                self.frame.flags.writeable = True
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                if self.results.multi_face_landmarks:
                    for face_landmarks in self.results.multi_face_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image=self.frame,
                            landmark_list=face_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())
                        self.mp_drawing.draw_landmarks(
                            image=self.frame,
                            landmark_list=face_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing_styles
                            .get_default_face_mesh_contours_style())
                        self.mp_drawing.draw_landmarks(
                            image=self.frame,
                            landmark_list=face_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing_styles
                            .get_default_face_mesh_iris_connections_style())


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@gzip.gzip_page
def facemesh(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        print("에러입니다....")
        pass
