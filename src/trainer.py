import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F

def train(gen, dis, sats, maps):
    adv_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()

    lr = 0.0002
    opti_g = optim.Adam(gen.parameters(), lr=lr)
    opti_d = optim.Adam(dis.parameters(), lr=lr)
    
    epochs = 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = gen.to(device=device)
    dis = dis.to(device=device)
    counter = 0
    for epoch in range(epochs):
        
        # each DataLoader yields a batch tensor (batch, C, H, W)
        for sat_img_batch, map_img_batch in zip(sats, maps):
            sat_img = sat_img_batch.to(device=device)
            map_img = map_img_batch.to(device=device)

            # print shapes for the first batch to help debug shape mismatches
            if counter == 0 and epoch == 0:
                print(f"DEBUG: sat_img shape={tuple(sat_img.shape)}, map_img shape={tuple(map_img.shape)}")
            fmap = gen(sat_img)

            # ensure generator output matches the target map size
            if fmap.shape[2:] != map_img.shape[2:]:
                fmap = F.interpolate(fmap, size=map_img.shape[2:], mode='bilinear', align_corners=False)

            # dis loss
            realL = adv_loss(dis(map_img), torch.ones_like(dis(map_img)))
            fakeL = adv_loss(dis(fmap.detach()), torch.zeros_like(dis(fmap)))
            d_loss = (realL + fakeL) / 2

            opti_d.zero_grad()
            d_loss.backward()
            opti_d.step()

            # Gen loss
            g_loss = adv_loss(dis(fmap), torch.ones_like(dis(fmap)))
            cycle_loss = l1_loss(fmap, map_img)
            total_g_loss = g_loss + 10 * cycle_loss

            opti_g.zero_grad()
            total_g_loss.backward()
            opti_g.step()
            counter += 1
            if counter % 10 ==0:
                print(f"Data: {counter}({epoch}/{epochs}) -- Gen Loss(total): {g_loss:.2f}({total_g_loss:.2f}) -- Dis Loss(Real - Fake)/2: {d_loss:.2f} (({realL:.2f} - {fakeL:.2f})/2)")
        
    pass
