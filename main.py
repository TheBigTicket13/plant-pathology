from prepare_data import load_csv, shuffle_data, split_data, load_images
from models import create_model
import cv2


if __name__ == "__main__":
    img_paths, one_hot_labels = load_csv()

    img_paths, one_hot_labels = shuffle_data(img_paths, one_hot_labels)
    img_train_paths, img_val_paths, img_test_paths = split_data(img_paths, test_split=0.15)
    one_hot_train, one_hot_val, one_hot_test = split_data(one_hot_labels, test_split=0.15)

    train_images, val_images, test_images = load_images((img_train_paths, img_val_paths, img_test_paths))
    '''
    for i in range(200,len(train_images)):
        #cv2.imshow("train_images", train_images[i])
        print(one_hot_train[i])
        cv2.waitKey(300)'''

    model = create_model()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", "mse"])
    # model train wypieprza sie jak mam wiecej obrazow jak 200 do 500, pozniej nie wiem
    # obrazy i labele wygladaja git
    # cos jest nie tak ze zbiorem treningowym ma zly shape zamiast (375, 351,512,3) jest (375,)
    model.fit(train_images, one_hot_train)







