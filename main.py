import prepare_training
import create_model
import predict_dataset
import os

if __name__ == "__main__":
    if not os.path.exists("ModelAntrenat.h5"):
        print("Modelul nu exista si va antrenat.")
        prepare_training.main()
        create_model.main()
    print("Modelul a fost deja antrenat.")
    predict_dataset.main()