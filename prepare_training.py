import os
import cv2
import json
import numpy as np

def process_and_extract_features(dataset_path, target_width=1536):
    data = []
    
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                if file.endswith('.jpg'):
                    file_path = os.path.join(label_path, file)
                    
                    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        continue
                    
                    height, width = image.shape
                    scal_factor = target_width / width
                    height_bun = int(height * scal_factor)
                    img_final = cv2.resize(image, (target_width, height_bun))
                    
                    _, thresh = cv2.threshold(img_final, 128, 255, cv2.THRESH_BINARY_INV)
                    
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    baseline = float(np.mean(img_final) / 255.0)
                    pen_pressure = float(np.sum(thresh > 0) / img_final.size) * 1000
                    
                    top_margin = 0
                    non_zero_rows = np.any(thresh > 0, axis=1)  # randuri cu pixeli activi
                    if np.any(non_zero_rows):
                        first_non_zero_row = np.argmax(non_zero_rows)  # prima linie cu text
                        top_margin = first_non_zero_row / img_final.shape[0]  # proportie
                    
                    letter_size = float(np.mean([cv2.boundingRect(c)[3] for c in contours if cv2.contourArea(c) > 5]))
                    
                    line_positions = sorted([cv2.boundingRect(c)[1] for c in contours if cv2.contourArea(c) > 5])
                    line_spacing = float(np.mean(np.diff(line_positions)) / scal_factor) if len(line_positions) > 1 else 0
                    
                    word_positions = sorted([cv2.boundingRect(c)[0] for c in contours if cv2.contourArea(c) > 5])
                    word_spacing = float(np.mean(np.diff(word_positions)) / scal_factor) if len(word_positions) > 1 else 0
                    
                    slant = float(np.degrees(np.mean([np.arctan(cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)[1]) for c in contours if len(c) > 5]))) if contours else 0
                    
                    data.append({
                        "label": label,
                        "file_path": file_path,
                        "features": {
                            "baseline": baseline,
                            "letter_size": letter_size,
                            "line_spacing": line_spacing,
                            "word_spacing": word_spacing,
                            "top_margin": top_margin,
                            "pen_pressure": pen_pressure,
                            "slant": slant
                        }
                    })
    
    return data


def main():
    dataset_path = "DateAntrenament"
    print("Extragem informatii din setul de date categorizat pe trasatura dominanta...")
    data = process_and_extract_features(dataset_path, target_width=1536)
    
    output_file = "proprietati_date_antrenament.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"Datele analizate au fost salvate in {output_file}")
