import os
import cv2
import numpy as np
import tensorflow as tf
import json

MODEL_PATH = "ModelAntrenat.h5"
IAM_DATASET_PATH = "IAM_Dataset/lines"
OUTPUT_JSON_PATH = "iam_predictions.json"

def preprocess_image(image_path, target_size=(7,)): # nr proprietati json
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Imagine invalidă: {image_path}")
        return None

    # normalizare
    baseline = np.mean(image[0:5, :])
    letter_size = np.mean(cv2.findNonZero(image)) if cv2.findNonZero(image) is not None else 0
    line_spacing = np.std(image) # spatierea liniilor
    word_spacing = np.mean(image) # spatiere cuvinte
    top_margin = np.mean(image[:10, :]) # margine sus
    pen_pressure = np.mean(image)
    slant = np.var(image) # inclinatie

    features = [
        baseline,
        letter_size,
        line_spacing,
        word_spacing,
        top_margin,
        pen_pressure,
        slant
    ]
    return np.array(features).reshape(1, -1) #date pt model

def classify_images(model, iam_dataset_path, output_json_path):
    limit = 10
    curr = 0
    results = []
    for root, _, files in os.walk(iam_dataset_path):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                curr = curr + 1
                if (curr > limit):
                    break
                image_path = os.path.join(root, file)
                features = preprocess_image(image_path)
                if features is None:
                    continue

                prediction = model.predict(features)
                scores = prediction.flatten()

                results.append({
                    "file_path": image_path,
                    "scores": {
                        "Agreeableness": float(scores[0]),
                        "Conscientiousness": float(scores[1]),
                        "Extraversion": float(scores[2]),
                        "Neuroticism": float(scores[3]),
                        "Openness": float(scores[4])
                    }
                })

    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Rezultatele sunt salvate in {output_json_path}")

def main():
    print("Incarcam modelul...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Clasificam imaginile...")
    classify_images(model, IAM_DATASET_PATH, OUTPUT_JSON_PATH)
