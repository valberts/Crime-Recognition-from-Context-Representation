import numpy as np


def cat_to_num(cat):
    categories = {
        "Abuse": 0,
        "Arrest": 1,
        "Arson": 2,
        "Assault": 3,
        "Burglary": 4,
        "Explosion": 5,
        "Fighting": 6,
        "RoadAccidents": 7,
        "Robbery": 8,
        "Shooting": 9,
        "Shoplifting": 10,
        "Stealing": 11,
        "Vandalism": 12,
    }

    return categories.get(cat)


def label2onehot(label):
    one_hot_label = [0] * 13
    one_hot_label[cat_to_num(label)] = 1
    return one_hot_label


if __name__ == "__main__":
    arr1 = np.load(
        "C:/dev/valberts/Crime-Recognition-from-Context-Representation/dataset/results/HRC_test_cat.npy"
    )
    arr2 = np.load(
        "C:/dev/valberts/Crime-Recognition-from-Context-Representation/dataset/results/HRC_train_cat.npy"
    )
    res1 = []
    res2 = []

    for e in arr1:
        res1.append(label2onehot(e))

    for e in arr2:
        res2.append(label2onehot(e))

    res1 = np.array(res1)
    res2 = np.array(res2)

    with open(
        "C:/dev/valberts/Crime-Recognition-from-Context-Representation/dataset/results/HRC_test_cat_oh.npy",
        "wb",
    ) as f:
        np.save(f, res1)

    with open(
        "C:/dev/valberts/Crime-Recognition-from-Context-Representation/dataset/results/HRC_train_cat_oh.npy",
        "wb",
    ) as f:
        np.save(f, res2)
