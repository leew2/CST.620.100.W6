import torch.optim as optim
import torch.nn as nn
import torch

def train(gen, dis, sats, maps):
    adv_loss = nn.BCELoss
    l1_loss = nn.L1Loss

    lr = 0.0002
    opti_g = optim.Adam(gen.parameter(), lr=lr)
    opti_d = optim.Adam(dis.parameter(), lr=lr)
    
    epochs = 10

    for epoch in range(epochs):
        counter = 0
        for (sat_img, _), (map_img, _) in zip(sats, maps):
            fmap = gen(sat_img)

            # dis loss
            realL = adv_loss(dis(map_img), torch.ones_like(dis(map_img)))
            fakeL = adv_loss(dis(fmap.detach()), torch.zeros_like(dis(fmap)))
            d_loss = (realL - fakeL)/2

            opti_d.zero_grad()
            d_loss.backward()
            opti_d.step()

            # Gen loss
            g_loss = adv_loss(dis(fmap), torch.ones_like(dis(fmap)))
            cycle_loss = l1_loss(fmap, map_img)
            total_g_loss = g_loss + 10 * cycle_loss

            opti_g.zero_grad()
            total_g_loss.backwards()
            opti_g.step()
            counter += 1
            if counter % 200 ==0:
                print(f"Data: {counter}({epoch}/{epochs}) -- Gen Loss(total): {g_loss}({total_g_loss}) -- Dis Loss((Real - Fake)/2: {d_loss} (({realL} - {fakeL})/2)")

    pass
