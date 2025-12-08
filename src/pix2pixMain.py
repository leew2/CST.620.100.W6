from data import *
from evaluate import *
from model import *
from trainer import *


def main():
    m, s = get_data()
    g, d = get_models()
    train(gen=g, dis=d, sats=s, maps=m)



    pass


if __name__=="__main__":
    main()
    print("Process Done")