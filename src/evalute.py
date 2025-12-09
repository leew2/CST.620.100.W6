import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch
from torchvision.transforms import ToPILImage
from model import get_models
from data import seprate_img


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gen, _ = get_models()
    gen = gen.to(device)
    gen.eval()

    # get a test satellite/map pair (seprate_img uses data/map/1.jpg by default)
    sat_t, map_t = seprate_img()

    # add batch dim and move to device
    x = sat_t.unsqueeze(0).to(device)

    with torch.no_grad():
        out = gen(x)

    # generator uses tanh -> scale to [0,1]
    out = (out + 1.0) / 2.0
    out = out.clamp(0.0, 1.0)

    toPIL = ToPILImage()
    img = toPIL(out.squeeze(0).cpu())
    sat = toPIL(sat_t)
    sat.show()
    
    os.makedirs('outputs', exist_ok=True)
    out_path = os.path.join('outputs', 'gen_test.jpg')
    img.save(out_path)

    print('Saved generated image to:', out_path)


if __name__ == '__main__':
    main()
