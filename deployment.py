import cv2
import numpy as np


def predict_condition(images, trained_model, label, spec_range):
    label_names = ["healthy", "multiple_diseases", "rust", "scab"]

    for i in range(spec_range[0], spec_range[1]):
        cv2.imshow(f"window {i}", images[i])
        prediction = trained_model.predict(np.reshape(images[i], (-1, 341, 512, 3)))
        print(f"Leaf nr {i} is/has ", label_names[np.argmax(prediction)],
              f" with {np.amax(prediction) * 100:.2f}% confidence.")
        print(f"This prediction is {np.argmax(prediction) == np.argmax(label[i])}.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()