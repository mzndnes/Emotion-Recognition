import os # operating system
import cv2 # open-cv package to train/predict
import glob # read/write file path
import random # randomize
import numpy # math package for array, number operations

import shutil # copy file operation, remove directory

class FacialEmotionRecognition:

    def __init__(self):
        self.emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] # list represents array in python
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
        self.fishface = cv2.face.FisherFaceRecognizer_create()

    # Organize given dataset based on emotion
    def pre_process_data(self):
        print("-------Pre-Processing Dataset------")

        if os.path.exists(self.sorted_emotion):
            print("%%%%% Removing existing path %s%%%%%" %(self.sorted_emotion))
            shutil.rmtree(self.sorted_emotion)

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

                    dest_neutral = "%s/%s" %(self.sorted_emotion, self.neutral)
                    if not os.path.exists(dest_neutral):
                        os.makedirs(dest_neutral)
                    dest_neutral_file = "%s/%s" %(dest_neutral, source_neutral_file[16:])

                    dest_emotion = "%s/%s" %(self.sorted_emotion, self.emotions[emotion_number])
                    if not os.path.exists(dest_emotion):
                        os.makedirs(dest_emotion)
                    dest_emotion_file = "%s/%s" %(dest_emotion, source_emotion_file[16:])

                    shutil.copyfile(source_neutral_file, dest_neutral_file)
                    shutil.copyfile(source_emotion_file, dest_emotion_file)


    def filter_multiple_neutral_faces_per_person(self):
        files = sorted(glob.glob("%s/%s/*" % (self.sorted_emotion, "neutral")))
        unique_images = set()
        for file in files:
            file_name = file.split("/")[-1].split("_")[0]
            if file_name in unique_images:
                os.remove(file)
            else:
                unique_images.add(file_name)


    def filter_faces_from_dataset(self):
        print("-------Filtering Faces from Dataset------")

        if os.path.exists(self.filtered_dataset):
            print("%%%%% Removing existing path %s%%%%%" %(self.filtered_dataset))
            shutil.rmtree(self.filtered_dataset)

        for emotion in self.emotions:

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
                        output_image = cv2.resize(gray_scale, (26551,300))
                        dest = "%s/%s" %(self.filtered_dataset, emotion)
                        if not os.path.exists(dest):
                            os.makedirs(dest)
                        cv2.imwrite("%s/%s.%s" %(dest, num, self.image_format), output_image)
                    except:
                        pass


    def train_predict_dataset_split(self, emotion):
        files = sorted(glob.glob("%s/%s/*" %(self.filtered_dataset, emotion)))
        random.shuffle(files)
        training_set = files[0:int(len(files)*0.01)]
        prediction_set = files[-int(len(files)*0.001):]
        return training_set, prediction_set


    def make_list_of_images_and_data_label(self, test_set, emotion):
        data_set = []
        label_set = []
        for item in test_set:
            img = cv2.imread(item)
            data_set.append(img)
            label_set.append(self.emotions.index(emotion))
        return data_set, label_set


    def generate_data_and_labels(self):
        print("-------Generating Training and Prediction Data from Dataset------")

        training_data_set = []
        training_label_set = []
        prediction_data_set = []
        prediction_label_set = []

        for emotion in self.emotions:
            print("-------Generating Training and Prediction Data for %s------", emotion)

            train_set, predict_set = self.train_predict_dataset_split(emotion)

            training_data, training_label = self.make_list_of_images_and_data_label(train_set, emotion)
            training_data_set += training_data
            training_label_set += training_label

            prediction_data, prediction_label = self.make_list_of_images_and_data_label(predict_set, emotion)
            prediction_data_set += prediction_data
            prediction_label_set += prediction_label

        return training_data_set,training_label_set, prediction_data_set, prediction_label_set


    def run_classifier(self):

        training_data, training_label, prediction_data, prediction_label = self.generate_data_and_labels()

        print("-------Training Fisher Face Classifier------")
        print("Size of training set: %s images" %(len(training_data)))

        self.fishface.train(training_data, numpy.array(training_label))

        print("-------Running Prediction------")
        print("Size of prediction set: %s images" % (len(prediction_data)))

        correct = 0
        incorrect = 0

        for ind, img in enumerate(prediction_data):
            prediction = self.fishface.predict(img)[0]
            if prediction == prediction_label[ind]:
                correct += 1
            else:
                incorrect += 1
        return (correct/ (correct + incorrect)) * 100


if __name__ == '__main__':
    number_of_runs = 1
    fer = FacialEmotionRecognition()
    fer.pre_process_data()
    fer.filter_multiple_neutral_faces_per_person() # efficiency step
    fer.filter_faces_from_dataset()

    accuracy_list = []

    file = open("results.txt", 'w')
    for i in range(number_of_runs):
        print("\n***********RUN %s*************\n" %(i+1))
        accuracy = fer.run_classifier()
        print("Classification Accuracy: %s" %(accuracy))
        accuracy_list.append(accuracy)
        file.write("Run %s accuracy: %s" %(i+1, accuracy))
    file.write("\nAvg accuracy: %s" %(numpy.mean(accuracy_list)))
    file.close()

    print("\n*******Total accuracy across %s runs: %s\n" %(number_of_runs, numpy.mean(accuracy_list)))
