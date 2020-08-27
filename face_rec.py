import face_recognition
import cv2
import nltk
import numpy as np
import time
import speech_recognition as sr
import pandas as pd
from gtts import gTTS
from threading import Thread
from multiprocessing.pool import ThreadPool
import os
from mpyg321.mpyg321 import  MPyg321Player
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')



def hear():
    rec = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening")
        aud = rec.listen(source,phrase_time_limit=5)
    try:
        sentence = rec.recognize_google(aud,language='en-US')
        tags = nltk.pos_tag(nltk.word_tokenize(sentence))
        name = [names for names, tag in tags if tag in ['NNP','JJ','RB']]
        print(tags)
        return name[0]
    except:
        return 'Unknown'







video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("/home/chakresh/Downloads/Modi.jpeg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]



# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,

]
known_face_names = [
    "Modi",

]


def face_recon():
    # Initialize some variables

    face_locations = []
    face_names = []
    data={}
    xl_encoding=[]
    process_this_frame = True
    global known_face_encodings , known_face_names
    player = MPyg321Player()
    already_recognized_face_names = []

    try:
        face_records = pd.read_excel('face_codes.xlsx')
        for cols in face_records:
            xl_encoding.append(np.array(face_records[cols]))

        xl_name = list(face_records.columns)
        known_face_encodings= xl_encoding
        known_face_names = xl_name

        print("4" ,known_face_encodings)
    except:
        pass


    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    if name not in already_recognized_face_names:
                        speak_name = gTTS(text="Hello " + name + " you are handsome", lang='en', slow=False)
                        speak_name.save('name.mp3')
                        player.play_song("name.mp3")
                        print(face_names, name)
                        already_recognized_face_names.append(name)
                else:
                    new_embedding = face_recognition.face_encodings(frame)[0]
                    speak_name = gTTS(text="Hello there, what's your name?", lang='en', slow=False)
                    speak_name.save('name.mp3')
                    player.play_song("name.mp3")
                    print("Enter : ")
                    time.sleep(4)

                    name = hear()
                    known_face_encodings.append(new_embedding)
                    known_face_names.append(name)

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            for keys in known_face_names:
                for values in known_face_encodings:
                    data[keys]=values
                    known_face_encodings.remove(values)
                    break
            print(data)
            df_k= pd.DataFrame(data)
            print("df",df_k)
            df_k.to_excel('face_codes.xlsx',index=False)

            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    face_recon()
