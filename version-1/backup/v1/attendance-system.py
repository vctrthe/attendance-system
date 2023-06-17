import os
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
import datetime

def preprocess_face(face):
    # Preprocess the face image if required (e.g., resizing, normalization)
    default_image_size = (1000, 1000)
    resize_default_size = cv2.resize(face, default_image_size)
    target_image_size = (160, 160)
    resize_target_size = cv2.resize(resize_default_size, target_image_size)
    normalized_face = resize_target_size / 255.0
    return normalized_face

def load_embeddings(person_folder):
    embeddings = []
    labels = []

    for person_file in os.listdir(person_folder):
        person_path = os.path.join(person_folder, person_file)
        if os.path.isfile(person_path) and person_path.endswith('_embeddings.pt'):
            print(f"Loading embeddings from {person_file}...")
            embedding_dict = torch.load(person_path)
            embeddings.append(embedding_dict['embeddings'])
            labels.append(person_file.split('_')[0])

    if embeddings:
        embeddings_tensor = torch.cat(embeddings, dim=0)
        labels_tensor = torch.tensor(labels)
        return embeddings_tensor, labels_tensor
    else:
        raise ValueError("No embeddings found.")

def recognize_person(embeddings, labels, face, mtcnn, facenet, threshold=0.7):
    face = preprocess_face(face)
    face = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        embedding = facenet(face)
        embedding = embedding.squeeze()

        distances = torch.norm(embeddings - embedding, dim=1)
        min_distance, min_index = torch.min(distances, dim=0)
        if min_distance < threshold:
            recognized_label = labels[min_index]
            return recognized_label
        else:
            return None

def mark_attendance(person_name, attendance_df):
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    attendance_df.loc[date_str, person_name] = time_str

def generate_attendance_report(attendance_df, save_path):
    attendance_df.to_excel(save_path)

def main():
    person_folder = 'face-embeddings'
    attendance_report_path = 'attendance_report.xlsx'
    threshold = 0.7

    # Load embeddings
    try:
        embeddings, labels = load_embeddings(person_folder)
    except ValueError as e:
        print(str(e))
        return

    # Initialize MTCNN and FaceNet models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Initialize GUI window
    window = tk.Tk()
    window.title("Attendance System")
    window.geometry("400x200")

    # Initialize attendance dataframe
    attendance_df = pd.DataFrame(columns=['Date/Time'])

    def capture_attendance():
        video_capture = cv2.VideoCapture(0)
        while True:
            ret, frame = video_capture.read()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(rgb_frame)

            if boxes is not None:
                for box in boxes:
                    x, y, w, h = box
                    cropped_face = frame[int(y):int(y + h), int(x):int(x + w)]
                    recognized_person = recognize_person(embeddings, labels, cropped_face, mtcnn, facenet, threshold)
                    if recognized_person:
                        mark_attendance(recognized_person, attendance_df)
                        messagebox.showinfo("Attendance Captured", f"Attendance marked for {recognized_person}.")

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
        generate_attendance_report(attendance_df, attendance_report_path)

    # Button to start capturing attendance
    start_button = tk.Button(window, text="Start Capturing Attendance", command=capture_attendance)
    start_button.pack(pady=20)

    window.mainloop()

if __name__ == '__main__':
    main()
