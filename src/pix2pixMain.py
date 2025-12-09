from data import *
from model import *
from trainer import *


def main():
    m, s = get_data()
    try:
        # if DataLoader has __len__, print sizes
        print(f"DEBUG: maps loader length={len(m)} sats loader length={len(s)}")
    except Exception:
        pass
    g, d = get_models()
    train(gen=g, dis=d, sats=s, maps=m)
    


    pass


if __name__=="__main__":
    main()
    print("Process Done")