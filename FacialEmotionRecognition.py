import glob
from shutil import copyfile

class FacialEmotionRecognition:

    def __init__(self):
        self.emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
        self.source_emotion = "emotion"
        self.source_image = "images"
        self.neutral = "neutral"
        self.sorted_emotion = "sorted_emotion_set"

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



if __name__ == '__main__':
    fer = FacialEmotionRecognition()
    fer.pre_process_data()