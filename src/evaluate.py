import matplotlib.pyplot as plt

def sample_img(dataset, name):
    img, _ = dataset[0]
    plt.imshow(img.permute(1,2,0))
    plt.title('Sample Image from: {name} dataset')


