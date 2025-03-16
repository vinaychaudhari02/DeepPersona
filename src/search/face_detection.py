import cv2
import os
import numpy as np
import logging
from insightface.app import FaceAnalysis

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize the InsightFace FaceAnalysis app (uses RetinaFace for detection and ArcFace for recognition)
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))  # Adjust detection size if needed

def detect_faces_and_get_embeddings(image_path, app):
    """
    Detect faces in an image and return a list of (embedding, bbox) tuples and the original image.
    If no faces are detected or the image cannot be read, returns an empty list and None.
    """
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Could not read image from {image_path}. Skipping file.")
        return [], None
    
    # Convert image to RGB as InsightFace expects images in RGB format
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb_image)
    
    if not faces:
        logging.info(f"No face detected in {image_path}")
        return [], image
    
    logging.info(f"Detected {len(faces)} face(s) in {image_path}")
    faces_info = []
    for face in faces:
        faces_info.append((face.embedding, face.bbox))
    return faces_info, image

# Process the reference image
reference_path = "/Users/vinaychaudhari/Documents/Face_Detection/citations.jpeg"  # Update with your reference image path
ref_faces, ref_image = detect_faces_and_get_embeddings(reference_path, app)
if not ref_faces:
    raise Exception("No face detected in the reference image. Please use an image with a clear face.")

if len(ref_faces) > 1:
    logging.info(f"Multiple faces detected in the reference image. Using all {len(ref_faces)} detected faces for matching.")
else:
    logging.info("One face detected in the reference image.")

# Optionally display the reference image with all detected faces marked
for embedding, bbox in ref_faces:
    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(ref_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
cv2.imshow("Reference Faces", ref_image)
cv2.waitKey(1000)  # display for 1 second
cv2.destroyAllWindows()

# Define the target directory containing multiple images
target_dir = "/Users/vinaychaudhari/Documents/Face_Detection/SearchEngine_Face_Detection/SearchEngine_Face_Detection/data/images/Harshil_Sharma__Saint_Louis/Bing/"  # Update with your target folder path

# Lowered cosine similarity threshold for matching (you can adjust further if needed)
cosine_threshold = 0.7

# Create a folder to save matched images if it doesn't exist
matched_dir = "matched_images"
if not os.path.exists(matched_dir):
    os.makedirs(matched_dir)
    logging.info(f"Created directory: {matched_dir}")

# Process each image in the target folder
for filename in os.listdir(target_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(target_dir, filename)
        target_faces, img = detect_faces_and_get_embeddings(image_path, app)
        # Skip the file if the image couldn't be read
        if img is None:
            continue
        
        # If no face detected, show the image and ask if it should be deleted
        if not target_faces:
            cv2.imshow("No Face Detected", img)
            cv2.waitKey(1000)  # display for 1 second
            cv2.destroyWindow("No Face Detected")
            answer = input(f"No face detected in {filename}. Delete this file? (y/n): ").strip().lower()
            if answer in ('y', 'yes'):
                os.remove(image_path)
                logging.info(f"Deleted image: {image_path}")
            else:
                logging.info(f"Kept image: {filename}")
            continue  # Skip further processing for this image

        # Check each detected face in the target image for a match
        match_found = False
        for (t_embedding, t_bbox) in target_faces:
            for idx, (r_embedding, _) in enumerate(ref_faces):
                cos_sim = np.dot(r_embedding, t_embedding) / (np.linalg.norm(r_embedding) * np.linalg.norm(t_embedding))
                logging.info(f"Comparing {filename}: Cosine similarity with reference face {idx+1} = {cos_sim:.2f}")
                if cos_sim > cosine_threshold:
                    logging.info(f"Match found in {filename} with cosine similarity: {cos_sim:.2f} (Reference face {idx+1})")
                    # Draw bounding box on the matched face
                    x1, y1, x2, y2 = t_bbox.astype(int)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"Sim: {cos_sim:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    match_found = True
                    break
            if match_found:
                break
        
        if match_found:
            # Save the matched image in the matched_images folder
            save_path = os.path.join(matched_dir, filename)
            cv2.imwrite(save_path, img)
            logging.info(f"Saved matched image: {save_path}")
            cv2.imshow("Matched Face", img)
            cv2.waitKey(1000)  # display for 1 second

cv2.destroyAllWindows()