from prepare_data import load_csv, load_single_image_path, load_some_images
import cv2
import keras
import numpy as np

if __name__ == "__main__":
    trained_model = keras.models.load_model("./best_model")
    img_paths, one_hot_labels = load_csv()
    images = load_some_images(img_paths, 15)

    for i in range(10):
        cv2.imshow(f"window {i}", images[i])
        print(f"Image nr {i}")
        prediction = trained_model.predict(np.reshape(images[i], (-1, 341, 512, 3)))
        np.argmax(prediction)
        cv2.waitKey(0)
        cv2.destroyAllWindows()










