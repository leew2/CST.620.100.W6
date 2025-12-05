import torchvision.transforms as transformer
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from evaluate import *

def get_data():
    # transformer
    transform = transformer.Compose([
        transformer.Resize((256, 256)),
        transformer.ToTensor(),
        transformer.Normalize((0.5,), (0.5,))
    ])

    # dataset
    sat_set = ImageFolder(root='./data/satellite', transform=transform)
    map_set = ImageFolder(root='./data/map', transform=transform)

    # loader
    size = 16
    sat_load = DataLoader(dataset=sat_set, batch_size=size, shuffle=True)
    map_load = DataLoader(dataset=map_set, batch_size=size, shuffle=True)

    sample_img(dataset=sat_set, name='Satellite')
    sample_img(dataset=map_set, name='Map')

    return sat_load, map_load


