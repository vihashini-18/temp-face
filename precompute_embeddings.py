import os
import numpy as np
import faiss
from tqdm import tqdm
from PIL import Image
from deepface import DeepFace
from src.utils import load_config
from src.utils import load_faiss_data, get_device


import faiss
import pickle
import os

def save_faiss_data(faiss_index, labels, config=None):
    # Use project src/ folder as default save path
    save_path = os.path.join(os.path.dirname(__file__))  # src/ folder

    # Save FAISS index
    faiss_file = os.path.join(save_path, "embeddings.faiss")
    faiss.write_index(faiss_index, faiss_file)
    
    # Save labels
    labels_file = os.path.join(save_path, "labels.pkl")
    with open(labels_file, "wb") as f:
        pickle.dump(labels, f)
    
    print(f"✅ FAISS index saved to {faiss_file}")
    print(f"✅ Labels saved to {labels_file}")



def precompute_embeddings():
    config = load_config()
    if not config:
        return

    dataset_dir = config['PATHS']['DATASET_DIR']
    embedding_model = config['RECOGNITION']['EMBEDDING_MODEL']
    
    
    print(f"Initializing DeepFace with model: {embedding_model}...")
    

    all_embeddings = []
    all_labels = []

    
    person_folders = [f.name for f in os.scandir(dataset_dir) if f.is_dir()]
    
    if not person_folders:
        print(f"No person folders found in {dataset_dir}. Check your folder structure.")
        return

    print(f"Found {len(person_folders)} persons to process.")

    for person_name in tqdm(person_folders, desc="Generating Embeddings"):
        person_dir = os.path.join(dataset_dir, person_name)
        image_files = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            print(f"Warning: No images found for {person_name}. Skipping.")
            continue

        
        
        person_embeddings = []
        
        for image_name in image_files:
            image_path = os.path.join(person_dir, image_name)
            try:
                
                representations = DeepFace.represent(
                    img_path=image_path, 
                    model_name=embedding_model, 
                    enforce_detection=False,
                    detector_backend='opencv',
                    # action='all' is an older parameter, just use defaults
                )
                
                
                if representations:
                    embedding = representations[0]['embedding']
                    person_embeddings.append(np.array(embedding))
                else:
                    print(f"Could not detect face in {image_path}. Skipping.")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        if person_embeddings:
            mean_embedding = np.mean(person_embeddings, axis=0)
            
            all_embeddings.append(mean_embedding)
            all_labels.append(person_name)


    
    if not all_embeddings:
        print("No embeddings were generated. FAISS index not created.")
        return

    embeddings_matrix = np.array(all_embeddings).astype('float32')
    dimension = embeddings_matrix.shape[1]
    
    
    
    print(f"Creating FAISS Index (Dimension: {dimension})...")
    faiss_index = faiss.IndexFlatL2(dimension)
    

    faiss_index.add(embeddings_matrix) 
    print(f"Total embeddings added to FAISS: {faiss_index.ntotal}")

    
    save_faiss_data(faiss_index, all_labels, config)


if __name__ == "__main__":
    precompute_embeddings()