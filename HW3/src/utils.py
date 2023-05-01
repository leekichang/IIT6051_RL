import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='RL HW3')
    parser.add_argument('--alpha' , default=0.3, type=float)
    parser.add_argument('--n_step', default=1, type=int)
    parser.add_argument('--n_tiles', default=8, type=int)
    args = parser.parse_args()
    return args