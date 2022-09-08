from train import N_gram
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix',
                        type=str,
                        default='',
                        help='Initial part of sequence')
    parser.add_argument('--model',
                        type=str,
                        help='Path to file where the model is saved')
    parser.add_argument('--length',
                        type=int,
                        help='Sequence length')
    args = parser.parse_args()

    n_gram = N_gram()
    n_gram.load(args.model)
    seed = 12
    n_gram.generate(args.length, seed, prefix=args.prefix)
