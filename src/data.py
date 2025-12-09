import torchvision.transforms as transformer
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import glob
from PIL import Image
import torch


def get_data():

    # dataset
    s_set , m_set = get_imgs()


    # loader
    size = 16
    sat_load = DataLoader(dataset=s_set, batch_size=size, shuffle=True)
    map_load = DataLoader(dataset=m_set, batch_size=size, shuffle=True)


    toImg = transformer.ToPILImage()
    sat = toImg(s_set[0])
    # Commented out to avoid opening an external image viewer and blocking execution
    # sat.show()

    return map_load, sat_load

def get_imgs():
    sats = []
    maps = []
    path = 'data/map'
    for img_paths in glob.glob(f'{path}/*.jpg'):
        img = Image.open(img_paths)
        s, m = seprate_img(img)
        sats.append(s)
        maps.append(m)
    return sats, maps

def seprate_img(img = Image.open('data/map/1.jpg')):
    w, h = img.size
    sat_img = img.crop((0, 0, w/2, h))
    map_img = img.crop((w/2, 0, w, h))

    

    sat_t = _transform(sat_img)
    map_t = _transform(map_img)

    return sat_t, map_t

def _transform(img):
    transform = transformer.Compose([
        transformer.Resize((256, 256)),
        transformer.ToTensor(),
    ])
    image = transform(img)
    return image


