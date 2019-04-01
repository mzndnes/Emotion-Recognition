import cv2
import glob
from shutil import copyfile

class FacialEmotionRecognition:

    def __init__(self):
        self.emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
        self.source_emotion = "emotion"
        self.source_image = "images"
        self.neutral = "neutral"
        self.sorted_emotion = "sorted_emotion_set"
        self.face_detection_filter_1 = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        self.face_detection_filter_2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
        self.face_detection_filter_3 = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")
        self.face_detection_filter_4 = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.filtered_dataset = "filtered_face_dataset"
        self.image_format = "jpg"

    def pre_process_data(self):
        volunteers = sorted(glob.glob("%s/*" %self.source_emotion))
        for volunteer in volunteers:
            volunteer_id = volunteer[-4:]
            for session in sorted(glob.glob("%s/*" %volunteer)):
                session_id = session[13:17]
                for file_name in sorted(glob.glob("%s/*"  %session)):
                    file = open(file_name, 'r')
                    emotion_number = int(float(file.readline()))
                    source_emotion_file = sorted(glob.glob("%s/%s/%s/*" %(self.source_image, volunteer_id, session_id)))[-1]
                    source_neutral_file = sorted(glob.glob("%s/%s/%s/*" %(self.source_image, volunteer_id, session_id)))[0]
                    dest_neutral_file = "%s/%s/%s" %(self.sorted_emotion, self.neutral, source_neutral_file[16:])
                    dest_emotion_file = "%s/%s/%s" %(self.sorted_emotion, self.emotions[emotion_number], source_emotion_file[16:])
                    copyfile(source_neutral_file, dest_neutral_file)
                    copyfile(source_emotion_file, dest_emotion_file)

    def filter_faces_from_dataset(self, emotion):
        files = sorted(glob.glob("%s/%s/*" %(self.sorted_emotion, emotion)))

        for num, file in enumerate(files, 0):
            image = cv2.imread(file)
            gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            face_1 = self.face_detection_filter_1.detectMultiScale(gray_scale, scaleFactor = 1.1, minNeighbors = 10,
                                                                   minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            face_2 = self.face_detection_filter_2.detectMultiScale(gray_scale, scaleFactor=1.1, minNeighbors=10,
                                                                   minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            face_3 = self.face_detection_filter_3.detectMultiScale(gray_scale, scaleFactor=1.1, minNeighbors=10,
                                                                   minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            face_4 = self.face_detection_filter_4.detectMultiScale(gray_scale, scaleFactor=1.1, minNeighbors=10,
                                                                   minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

            if len(face_1) == 1:
                face_features = face_1
            elif len(face_2) == 1:
                face_features = face_2
            elif len(face_3) == 1:
                face_features = face_3
            elif len(face_4) == 1:
                face_features = face_4
            else:
                face_features = ""

            for (x, y, w, h) in face_features:
                gray_scale = gray_scale[y:y+h, x:x+w]
                try:
                    output_image = cv2.resize(gray_scale, (350, 350))
                    cv2.imwrite("%s/%s/%s.%s" %(self.filtered_dataset, emotion, num, self.image_format), output_image)
                except:
                    pass





if __name__ == '__main__':
    fer = FacialEmotionRecognition()
    # fer.pre_process_data()
    for emotion in fer.emotions:
        fer.filter_faces_from_dataset(emotion)