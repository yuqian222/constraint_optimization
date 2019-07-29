import argparse

def get_args():
    parser = argparse.ArgumentParser(description='cheetah_q parser')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')

    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')

    parser.add_argument('--render', action='store_true', default=False,
                        help='render the environment')

    parser.add_argument('--env', type=str, default='HalfCheetah-v2',
                        help='enviornment (default: HalfCheetah-v2)')

    parser.add_argument('--policy', type=str, default="nn",
                        help='policy trained. can be or linear or nn')

    parser.add_argument('--branches', type=int, default=10, metavar='N',
                        help='branches per round (default: 5)')

    parser.add_argument('--iter_steps', type=int, default=10000, metavar='N',
                        help='num steps per iteration (default: 10,000)')

    parser.add_argument('--var', type=float, default=0.05,
                        help='sample variance (default: 0.05)')

    parser.add_argument('--hidden_size', type=int, default=24,
                        help='hidden size of policy nn (default: 24)')

    parser.add_argument('--correct', type=bool, default=False,
                        help='whether to explore corrective actions or not')

    parser.add_argument('--training_epoch', type=int, default=500,
                        help='Training epochs for each policy update (default: 500)')
    
    parser.add_argument('--load_dir', type=str, default="")
    parser.add_argument('--n_samples', type=int, default=0)
    args = parser.parse_args()
    return args