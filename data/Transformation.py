import numpy as np
import torchvision.transforms as T
from imgaug import augmenters as iaa

class ImgAugTransformation:
    """
    Wrapper to allow imgaug to work with pytorch transformation pipeline
    """

    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.2, iaa.GaussianBlur(sigma =(0,3.0))),
            iaa.Sometimes(0.8, iaa.Affine(rotate = (-30,30), mode = 'symmetric')),
            iaa.Sometimes(0.5, iaa.Crop(percent = (0.1, 0.1), keep_size = True))
        ])

    def __call__(self, img):
        # array and reshape
        img = np.array(img).reshape((28,28))/255
        # augment image
        img = self.aug.augment_image(img)
        return img


train_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((28,28)),
    ImgAugTransformation(),
    T.ToTensor()
])

val_transform = T.Compose([
    T.Resize((28,28)),
    T.ToTensor()
])

test_transform = T.Compose([
    T.Resize((28,28)),
    T.ToTensor()
])
