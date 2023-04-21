import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='RL HW2')
    parser.add_argument('--n_state', default=5, type=int)
    parser.add_argument('--n_step', default=1, type=int)
    args = parser.parse_args()
    return args