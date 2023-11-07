import argparse

def get_args():
    args = argparse.ArgumentParser()
    # common
    args.add_argument("--tpose", type=str, help="T-pose path", default="Tpose/T-pos-normalize.pkl")
    args.add_argument("-p", "--prefix", type=str, help="Model path prefix", default="test")
    args.add_argument("--inp_len", type=int, help="Input length", default=20)
    args.add_argument("--out_len", type=int, help="Output length", default=10)
    args.add_argument("-V", "--version", type=str, help="Joint definition", default="V3")
    args.add_argument("-m", "--model", type=str, help="Model path", default="ckpt/")
    args.add_argument("-d", "--dataset", type=str, help="Dataset directory", default="../Dataset/Choreomaster")
    args.add_argument("-v", "--visual", help="Visualize", action="store_true")
    
    # training
    args.add_argument("--epochs", type=int, default=250)
    args.add_argument("--lr", type=float, help="learning rate", default=0.0001)
    args.add_argument("--batch_size", type=int, default=128)
    args.add_argument("-r", "--train_dir", type=str, help="Train directory", default="train_angle")
    args.add_argument("-c", "--train_ca", type=str, help="Train class", default="01")
    
    # Inference
    args.add_argument("-t", "--type", type=str, help="Inference type: infill/concat/smooth", default="smooth")
    args.add_argument("-f", "--file", type=str, help="File(pkl/xml) path")
    args.add_argument("-s", "--save", type=str, help="save path", default='result/demo')
    
    return args.parse_args()