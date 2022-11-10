from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading
import time
import mediapipe as mp


lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
lipsLowerInner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

silhouette = [10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
              397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
              172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109]

rightEyeUpper0 = [246, 161, 160, 159, 158, 157, 173]
rightEyeLower0 = [33, 7, 163, 144, 145, 153, 154, 155, 133]
rightEyeUpper1 = [247, 30, 29, 27, 28, 56, 190]
rightEyeLower1 = [130, 25, 110, 24, 23, 22, 26, 112, 243]
rightEyeUpper2 = [113, 225, 224, 223, 222, 221, 189]
rightEyeLower2 = [226, 31, 228, 229, 230, 231, 232, 233, 244]
rightEyeLower3 = [143, 111, 117, 118, 119, 120, 121, 128, 245]

rightEyeUpper = [246, 161, 160, 159, 158, 157, 173,
                 247, 30, 29, 27, 28, 56, 190,
                 113, 225, 224, 223, 222, 221, 189]
rightEyeLower = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                 130, 25, 110, 24, 23, 22, 26, 112, 243,
                 226, 31, 228, 229, 230, 231, 232, 233, 244,
                 143, 111, 117, 118, 119, 120, 121, 128, 245]

rightEyebrowUpper = [156, 70, 63, 105, 66, 107, 55, 193]
rightEyebrowLower = [35, 124, 46, 53, 52, 65]

rightEyeIris = [473, 474, 475, 476, 477]

leftEyeUpper0 = [466, 388, 387, 386, 385, 384, 398]
leftEyeLower0 = [263, 249, 390, 373, 374, 380, 381, 382, 362]
leftEyeUpper1 = [467, 260, 259, 257, 258, 286, 414]
leftEyeLower1 = [359, 255, 339, 254, 253, 252, 256, 341, 463]
leftEyeUpper2 = [342, 445, 444, 443, 442, 441, 413]
leftEyeLower2 = [446, 261, 448, 449, 450, 451, 452, 453, 464]
leftEyeLower3 = [372, 340, 346, 347, 348, 349, 350, 357, 465]

leftEyeUpper = [466, 388, 387, 386, 385, 384, 398,
                467, 260, 259, 257, 258, 286, 414,
                342, 445, 444, 443, 442, 441, 413]
leftEyeLower = [263, 249, 390, 373, 374, 380, 381, 382, 362,
                359, 255, 339, 254, 253, 252, 256, 341, 463,
                446, 261, 448, 449, 450, 451, 452, 453, 464,
                372, 340, 346, 347, 348, 349, 350, 357, 465]

leftEyebrowUpper = [383, 300, 293, 334, 296, 336, 285, 417]
leftEyebrowLower = [265, 353, 276, 283, 282, 295]

leftEyeIris = [468, 469, 470, 471, 472]


def min(val_a, val_b):
    if val_a > val_b:
        return val_b
    else:
        return val_a


def max(val_a, val_b):
    if val_a > val_b:
        return val_a
    else:
        return val_b


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

        # print('1. not address')
        # try:
        #     self.left_eye = self.results.face_mesh.landmark[33]
        #     self.right_eye = self.results.face_mesh.landmark[63]
        #     self.nose = self.results.face_mesh.landmark[94]
        #     self.mouth = self.results.face_mesh.landmark[13]

        # self.img = np.zeros((384,384,3), np.uint8)
        # for i in range(0, len(fonts)):
        #     self.point = 30,30+(i*40)
        #     cv2.putText(self.img, 'PYTHON', self.point, fonts[i],1,white_color,2,cv2.LINE_AA)

    ftr = 0
    ftl = 0
    yasl = 0
    yasr = 0
    yay = 0
    yaa = 0

    prevtime = 0

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
                self.curtime = time.time()
                self.sec = self.curtime - self.prevtime
                self.prevtime = self.curtime
                self.fps = 1 / (self.sec)
                self.str1 = "FPS : %0.1f" % self.fps
                self.frame = cv2.putText(
                    self.frame, self.str1, (0, 50), cv2.FONT_ITALIC, 1, (0, 255, 0))

                self.frame.flags.writeable = True
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
                # if self.results.multi_face_landmarks:
                #     for face_landmarks in self.results.multi_face_landmarks:
                #         self.mp_drawing.draw_landmarks(
                #             image=self.frame,
                #             landmark_list=face_landmarks,
                #             connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                #             landmark_drawing_spec=None,
                #             connection_drawing_spec=self.mp_drawing_styles
                #             .get_default_face_mesh_tesselation_style())
                #         self.mp_drawing.draw_landmarks(
                #             image=self.frame,
                #             landmark_list=face_landmarks,
                #             connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                #             landmark_drawing_spec=None,
                #             connection_drawing_spec=self.mp_drawing_styles
                #             .get_default_face_mesh_contours_style())
                #         self.mp_drawing.draw_landmarks(
                #             image=self.frame,
                #             landmark_list=face_landmarks,
                #             connections=self.mp_face_mesh.FACEMESH_IRISES,
                #             landmark_drawing_spec=None,
                #             connection_drawing_spec=self.mp_drawing_styles
                #             .get_default_face_mesh_iris_connections_style())

                def findMinMaxPos(face_part):

                    # extract face min max
                    min_x = 10000
                    max_x = -10000
                    min_y = 10000
                    max_y = -10000

                    #print('face_part :{}'.format(face_part))

                    for part_idx in face_part:
                        #print('part id: {}'.format(part_idx))
                        sh_x = self.results.multi_face_landmarks[0].landmark[part_idx].x
                        sh_y = self.results.multi_face_landmarks[0].landmark[part_idx].y
                        sh_z = self.results.multi_face_landmarks[0].landmark[part_idx].z

                        min_x = min(sh_x, min_x)
                        min_y = min(sh_y, min_y)

                        max_x = max(sh_x, max_x)
                        max_y = max(sh_y, max_y)

                    #print('Face shil min({},{}), max({},{})'.format(min_x, min_y, max_x, max_y))

                    return (min_x, min_y), (max_x, max_y)

                # extract face min max
                sil_min, sil_max = findMinMaxPos(silhouette)
                #print('Face shil min({},{}), max({},{})'.format(sil_min[0], sil_min[1], sil_max[0], sil_max[1]))

                r_eyeb_min, r_eyeb_max = findMinMaxPos(rightEyebrowUpper)
                l_eyeb_min, l_eyeb_max = findMinMaxPos(leftEyebrowUpper)
                r_eyeu_min, r_eyeu_max = findMinMaxPos(leftEyeUpper0)
                l_eyeu_min, l_eyeu_max = findMinMaxPos(rightEyeUpper0)

                r_eyeu0_min, r_eyeu0_max = findMinMaxPos(
                    rightEyeUpper0)  # 159 #r_eyeu0_max
                l_eyel0_min, l_eyel0_max = findMinMaxPos(
                    leftEyeLower0)  # 374 # l_eyel0_min

                l_eyeu0_min, l_eyeu0_max = findMinMaxPos(
                    leftEyeUpper0)  # 386 # l_eyeu0_max
                r_eyel0_min, r_eyel0_max = findMinMaxPos(
                    rightEyeLower0)  # 145 # r_eyel0_min

                lips_upi_min, lips_upi_max = findMinMaxPos(
                    lipsUpperInner)  # 13
                # 최소값 x[0],y[1] / 최대값 x[0],y[1]
                # turn right
                mark14 = self.results.multi_face_landmarks[0].landmark[14].y
                mark204 = self.results.multi_face_landmarks[0].landmark[204].y
                mark424 = self.results.multi_face_landmarks[0].landmark[424].y

                mark4 = self.results.multi_face_landmarks[0].landmark[4].y
                mark266 = self.results.multi_face_landmarks[0].landmark[266].y
                mark36 = self.results.multi_face_landmarks[0].landmark[36].y

                mark145 = self.results.multi_face_landmarks[0].landmark[145].y
                mark443 = self.results.multi_face_landmarks[0].landmark[443].y

                mark374 = self.results.multi_face_landmarks[0].landmark[374].y
                mark223 = self.results.multi_face_landmarks[0].landmark[223].y

                str1 = ""
                # cv2.putText(image, str1, (100,400), cv2.FONT_ITALIC, 1, (0, 255, 0))
                str2 = ""
                # cv2.putText(image, "warning", (250,200), cv2.FONT_ITALIC, 1, (255, 0, 0))
                # cv2.putText(image, "attention", (250,225), cv2.FONT_ITALIC, 1, (255, 255, 0))

                if (sil_min[0] < r_eyeb_min[0]):
                    ftr = 0

                if (sil_min[0] > r_eyeb_min[0]):
                    # print('Face turn right!')
                    str1 = "Face turn right!"
                    ftr += 1
                    print("ftr:"+str(ftr))
                    cv2.putText(self.frame, str1, (100, 350),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)
                    if (ftr >= 5):
                        str2 = "attention"
                    if (ftr > 20):
                        str2 = "attention"
                    if (ftr >= 50):
                        str2 = "warning"
                    if (str2 == "attention"):
                        cv2.putText(self.frame, str2, (250, 250),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 3)
                    if (str2 == "warning"):
                        cv2.putText(self.frame, str2, (250, 250),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 3)

                # turn left

                if (sil_max[0] > l_eyeb_max[0]):
                    ftl = 0

                if (sil_max[0] < l_eyeb_max[0]):
                    # print('Face turn left!')
                    str1 = "Face turn left!"
                    ftl += 1
                    print("ftl:"+str(ftl))
                    cv2.putText(self.frame, str1, (100, 350),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)
                    if (ftl >= 5):
                        str2 = "attention"
                    if (ftl > 20):
                        str2 = "attention"
                    if (ftl >= 50):
                        str2 = "warning"
                    if (str2 == "attention"):
                        cv2.putText(self.frame, str2, (250, 250),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 3)
                    if (str2 == "warning"):
                        cv2.putText(self.frame, str2, (250, 250),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 3)

                if (mark145 >= mark443):
                    yasl = 0

                # if(r_eyeu0_max[1] <l_eyel0_min[1]) :  #if (r_eyeu_max[1] > l_eyeu_max[1]):
                if (mark145 < mark443):
                    # print('you are sleeping left')
                    str1 = "you are sleeping right"
                    yasl += 1
                    print("yasl:" + str(yasl))
                    cv2.putText(self.frame, str1, (100, 400),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)
                    if (yasl >= 5):
                        str2 = "attention"
                    if (yasl > 20):
                        str2 = "attention"
                    if (yasl >= 50):
                        str2 = "warning"
                    if (str2 == "attention"):
                        cv2.putText(self.frame, str2, (250, 250),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 3)
                    if (str2 == "warning"):
                        cv2.putText(self.frame, str2, (250, 250),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 3)

                if (mark374 >= mark223):
                    yasr = 0

                # if(l_eyeu0_max[1] < r_eyel0_min[1]):   #if (r_eyeu_max[1] < l_eyeu_max[1]):
                if (mark374 < mark223):
                    # print('you are sleeping right')
                    str1 = "you are sleeping left"
                    yasr += 1
                    print("yasr:"+str(yasr))
                    cv2.putText(self.frame, str1, (100, 400),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)
                    if (yasr >= 5):
                        str2 = "attention"
                    if (yasr >= 50):
                        str2 = "warning"
                    if (str2 == "attention"):
                        cv2.putText(self.frame, str2, (250, 250),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 3)
                    if (str2 == "warning"):
                        cv2.putText(self.frame, str2, (250, 250),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 3)

                if (mark14 >= mark204 and mark14 >= mark424):
                    yay = 0

                # 하품하면   #if (mark13 <= mark436 and mark13 <= mark216):   #if (mark13 <= mark92 and mark13 <= mark322): #if (mark13 <= mark270 and mark13 <= mark40):
                if (mark14 > mark204 and mark14 > mark424):
                    # print('you are yawning')
                    str1 = "you are yawning"
                    yay += 1
                    print("yay:"+str(yay))
                    cv2.putText(self.frame, str1, (100, 400),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)
                    if (yay >= 5):
                        str2 = "attention"
                    if (yay > 20):
                        str2 = "attention"
                    if (yay >= 50):
                        str2 = "you need some rest"
                    if (str2 == "attention"):
                        cv2.putText(self.frame, str2, (250, 250),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 3)
                    if (str2 == "warning"):
                        cv2.putText(self.frame, str2, (250, 250),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 3)

                if (mark4 <= mark266 or mark4 <= mark36):
                    yaa = 0

                if (mark4 > mark266 and mark4 > mark36):  # 고개를 숙이면
                    # print('you are asleep')
                    str1 = "you are asleep"
                    yaa += 1
                    print("yaa:"+str(yaa))
                    cv2.putText(self.frame, str1, (100, 450),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)
                    if (yaa >= 5):
                        str2 = "attention"
                    if (yaa > 20):
                        str2 = "attention"
                    if (yaa >= 50):
                        str2 = "warning"
                    if (str2 == "attention"):
                        cv2.putText(self.frame, str2, (250, 250),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 3)
                    if (str2 == "warning"):
                        cv2.putText(self.frame, str2, (250, 250),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 3)

                r_eye_ir_min, r_eye_ir_max = findMinMaxPos(rightEyeIris)
                l_eye_ir_min, l_eye_ir_max = findMinMaxPos(leftEyeIris)

                r_ir_hgt = abs(r_eye_ir_max[1] - r_eye_ir_min[1])
                l_ir_hgt = abs(l_eye_ir_max[1] - l_eye_ir_min[1])
                """
            r_eye_up_min, r_eye_up_max = findMinMaxPos(rightEyeUpper)
            r_eye_low_min, r_eye_low_max = findMinMaxPos(rightEyeLower)
            
            l_eye_up_min, l_eye_up_max = findMinMaxPos(leftEyeUpper)
            l_eye_low_min, l_eye_low_max = findMinMaxPos(leftEyeLower)
            """
                r_eye_up_min, r_eye_up_max = findMinMaxPos(rightEyeUpper1)
                r_eye_low_min, r_eye_low_max = findMinMaxPos(rightEyeLower1)

                l_eye_up_min, l_eye_up_max = findMinMaxPos(leftEyeUpper1)
                l_eye_low_min, l_eye_low_max = findMinMaxPos(leftEyeLower1)

                r_eye_hgt = abs(r_eye_up_max[1] - r_eye_low_min[1])
                l_eye_hgt = abs(l_eye_up_max[1] - l_eye_low_min[1])

                # # bilnk_left
                # if l_eye_hgt < l_ir_hgt * 0.1:
                #     # if l_eye_hgt < l_ir_hgt * 0.05:
                #     print('blink left eye!( {}, {})'.format(l_eye_hgt, l_ir_hgt))

                # # blink_right
                # if r_eye_hgt < r_ir_hgt * 0.1:
                #     # if r_eye_hgt < r_ir_hgt * 0.05:
                #     print('blink right eye!( {}, {})'.format(r_eye_hgt, r_ir_hgt))


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
