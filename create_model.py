import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json

def balansare_copiere(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    # Organize features by label
    labels_dict = {}
    for item in data:
        label = item["label"]
        if label not in labels_dict:
            labels_dict[label] = []
        labels_dict[label].append([
            item["features"]["baseline"],
            item["features"]["letter_size"],
            item["features"]["line_spacing"],
            item["features"]["word_spacing"],
            item["features"]["top_margin"],
            item["features"]["pen_pressure"],
            item["features"]["slant"]
        ])

    max_size = max(len(features) for features in labels_dict.values())

    balanced_features = []
    balanced_labels = []
    for label, features in labels_dict.items():
        while len(features) < max_size:
            features.extend(features[:max_size - len(features)])
        balanced_features.extend(features[:max_size])
        balanced_labels.extend([label] * max_size)

    return np.array(balanced_features), np.array(balanced_labels)

def build_ann_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape), # dim de iesire
        tf.keras.layers.Rescaling(1.0 / 255), # normalizam intrarile la [0,1]
        tf.keras.layers.Dense(256, activation="relu"), # strat dens cu 256 noduri
        tf.keras.layers.Dropout(0.5),# reducem overfitting prin omiterea unor noduri
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax") # strat de iesire
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), # adam cu rata de invatare mai mica
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), # fct pierdere pt clasificarea multiclasa
        metrics=["accuracy"]
    )
    return model

def main():
    json_file = "proprietati_date_antrenament.json"
    print("Incarcam si fixam informatiile...")
    features, labels = balansare_copiere(json_file)

    print("Folosim LabelEncoder pentru a imbunatatii structurarea...")
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    class_names = label_encoder.classes_

    print("Clasele LabelEncoder:", class_names)

    print("Impartim datasetul folosind train_test_split din sklearn...")
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels_encoded, test_size=0.3, random_state=42, stratify=labels_encoded)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print("Construim modelul ANN...")
    input_shape = X_train.shape[1:]
    num_classes = len(class_names)
    model = build_ann_model(input_shape, num_classes)

    print("Antrenam modelul...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150, # nr mai mare de epoci
        batch_size=16, # impartirea datelor
        verbose=1 # progress antrenare
    )

    print("Testam modelul...")
    y_pred = np.argmax(model.predict(X_test), axis=1) # etichetele pt datele de test
    print("PRECIZIE MODEL:", np.mean(y_pred == y_test))

    print("\nReport clasificare din sklearn:\n", classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=1
    ))
    
    model.save("ModelAntrenat.h5") #todo: foloseste .keras

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names) # randuri: clase reale, coloane: clase prezise
    disp.plot(cmap="viridis")
    plt.show()
