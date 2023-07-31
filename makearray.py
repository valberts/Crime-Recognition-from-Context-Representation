import numpy as np

cats = [
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
]

cats = np.array(cats)
np.save(
    "C:/dev/valberts/Crime-Recognition-from-Context-Representation/high-level-context-representation/dataset/HRC_preprocessed/HRC_test_cat.npy",
    cats,
)
