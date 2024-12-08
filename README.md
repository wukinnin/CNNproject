# Cats VS Dogs image classification using CNN

## Requirements:

- Python 3
- `pip install numpy opencv-python tensorflow tflearn tensorflow-gpu`

---

Expected file format:
```

ccn-train.py
ccn-test.py
/test
    /cat
        cat0-test.jpg
        cat1-test.jpg
        [...]
    /dog
        dog0-test.jpg
        dog1-test.jpg
        [...]
/train
    /cat
        cat0.jpg
        cat1.jpg
        [...]
    /dog
        dog0.jpg
        dog1.jpg
        [...]
```

- Install all dependencies using pip.
- Run `ccn-train.py` to train the model and save it as animals_cnn.h5.
- Run `ccn-test.py` to classify a sample test image.
