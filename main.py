from prepare_data import load_csv, load_some_images
from deployment import predict_condition
import keras

if __name__ == "__main__":
    trained_model = keras.models.load_model("./best_model")
    img_paths, one_hot_labels = load_csv()
    images = load_some_images(img_paths, 25)

    predict_condition(images, trained_model, one_hot_labels, (15, 25))










