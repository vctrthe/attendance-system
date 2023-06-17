import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2

def preprocess_face(face):
    # Preprocess the face image if required (e.g., resizing, normalization)
    default_image_size = (1000, 1000)
    resize_default_size = cv2.resize(face, default_image_size)
    target_image_size = (160, 160)
    resize_target_size = cv2.resize(resize_default_size, target_image_size)
    normalized_face = resize_target_size / 255.0
    return normalized_face

def generate_embeddings(person_folder, save_folder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Limit System Resources
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    embeddings = []
    for person in os.listdir(person_folder):
        person_path = os.path.join(person_folder, person)
        if os.path.isdir(person_path):
            print(f"Processing person: {person}")
            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)
                face = cv2.imread(image_path)
                face = preprocess_face(face)
                face = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).to(device).float()

                embedding = facenet(face)
                if embedding.ndim > 1:
                    embeddings.append(embedding)

                # Cap RAM usage if necessary
                current_ram_usage = torch.cuda.max_memory_allocated()
                if current_ram_usage > 4e9:  # 4 GB
                    torch.cuda.empty_cache()

    if embeddings:
        embeddings_tensor = torch.cat(embeddings)
        save_path = os.path.join(save_folder, 'embeddings.pt')
        torch.save(embeddings_tensor, save_path)
        print(f"Embeddings saved to {save_path}")
    else:
        print("No embeddings generated.")

def main():
    person_folder = 'face-pictures/'
    save_folder = 'face-embeddings/'
    generate_embeddings(person_folder, save_folder)

if __name__ == '__main__':
    main()
