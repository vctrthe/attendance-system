import os
import torch
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import pandas as pd
import datetime
from tkinter import *
from tkinter import messagebox
import torch.nn.functional as F

def preprocess_face(face):
    # Preprocess the face image if required (e.g., resizing, normalization)
    default_image_size = (1000, 1000)
    resize_default_size = cv2.resize(face, default_image_size)
    target_image_size = (160, 160)
    resize_target_size = cv2.resize(resize_default_size, target_image_size)
    normalized_face = resize_target_size / 255.0
    return normalized_face

def load_embeddings(embeddings_file):
    loaded_embeddings = torch.load(embeddings_file)

    # Get the expected embedding size by passing a sample face through the model
    sample_face = torch.randn(1, 3, 160, 160).to(device)
    with torch.no_grad():
        embeddings = facenet(sample_face)
    expected_embedding_size = embeddings.size(-1)

    # Resize embeddings to match the expected size (if necessary)
    for person, embedding in loaded_embeddings.items():
        print(embedding.shape)  # Print the shape of the embedding
        if len(embedding.shape) == 2:
            embedding = embedding.permute(1, 0)  # Reshape to (C, N)
            embedding = F.interpolate(embedding.unsqueeze(0).unsqueeze(0), size=(expected_embedding_size, expected_embedding_size), mode='bilinear', align_corners=False)
            embedding = embedding.squeeze(0).squeeze(0).permute(1, 0)  # Reshape back to (N, C)
        elif len(embedding.shape) == 3:
            embedding = embedding.permute(2, 0, 1)  # Reshape to (C, d1, d2)
            embedding = F.interpolate(embedding.unsqueeze(0), size=(expected_embedding_size, expected_embedding_size), mode='trilinear', align_corners=False)
            embedding = embedding.squeeze(0).permute(1, 2, 0)  # Reshape back to (d1, d2, C)
        elif len(embedding.shape) == 1:
            embedding = embedding.unsqueeze(1)  # Reshape to (N, 1)
        else:
            raise ValueError(f"Invalid embedding shape: {embedding.shape}")

        loaded_embeddings[person] = embedding

    print(loaded_embeddings.keys())  # Print the keys of loaded embeddings
    return loaded_embeddings

def recognize_face(face, embeddings, facenet):
    face = preprocess_face(face)
    face = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).to(device).float()
    embedding = facenet(face)

    min_distance = float('inf')
    recognized_person = None

    for person, saved_embedding in embeddings.items():
        distance = torch.dist(embedding, saved_embedding)
        if distance < min_distance:
            min_distance = distance
            recognized_person = person
    
    if recognized_person is not None:
        return recognized_person
    else:
        return "UNKNOWN"

def log_attendance(person, attendance_log):
    now = datetime.datetime.now()
    timestamp = now.strftime("%d-%m-%Y %H:%M:%S")
    attendance_log.append((person, "Present", timestamp))

def save_attendance_log(attendance_log):
    now = datetime.datetime.now()
    output_path = f"attendance_log_{now.strftime('%d-%m-%Y')}.xlsx"
    df = pd.DataFrame(attendance_log, columns=["Name", "Status", "Timestamp"])
    df.to_excel(output_path, index=False)
    print(f"Attendance log saved to {output_path}")

def start_attendance_system():
    video_capture = cv2.VideoCapture(0)
    attendance_log = []

    while True:
        ret, frame = video_capture.read()

        # Detect faces using MTCNN
        boxes, _ = mtcnn.detect(frame)

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x, y, w, h = box
                face_img = frame[int(y):int(y+h), int(x):int(x+w)]

                # Recognize face using FaceNet embeddings
                person = recognize_face(face_img, embeddings, facenet)
                if person is not None:
                    print(f"Recognized person: {person}")
                    log_attendance(person, attendance_log)

                # Display the frame with bounding boxes and recognized person
                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
                cv2.putText(frame, person, (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.imshow('Attendance System', frame)

        # Stop the attendance system when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and destroy windows
    video_capture.release()
    cv2.destroyAllWindows()

    # Ask for confirmation to save the attendance log
    if messagebox.askyesno("Save Attendance Log", "Do you want to save the attendance log?"):
        save_attendance_log(attendance_log)
    else:
        print("Attendance log not saved.")

def stop_attendance_system():
    attendance_log = []

    for person in embeddings.keys():
        attendance_log.append((person, "Absent", ""))

    save_attendance_log(attendance_log)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MTCNN and FaceNet models
mtcnn = MTCNN(keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load embeddings
embeddings_file = 'face-embeddings/embeddings.pt'
embeddings = load_embeddings(embeddings_file)

# Create GUI
root = Tk()
root.title("Attendance System")

start_button = Button(root, text="Start Attendance", command=start_attendance_system)
start_button.pack(pady=10)

stop_button = Button(root, text="Stop Recording", command=stop_attendance_system)
stop_button.pack(pady=10)

root.mainloop()
