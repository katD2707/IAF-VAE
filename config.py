import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="train ResnetVAE model")
    parser.add_argument(
        "-ckpt",
        "--current_checkpoint",
        default="None",
        type=str,
    )
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--image_size', default=128, type=int)
    parser.add_argument('--in_channels', default=3, type=int)
    parser.add_argument('--hidden_size', default=160, type=int)
    parser.add_argument('--z_size', default=32, type=int)
    parser.add_argument('--k', default=1, type=int)
    parser.add_argument('--kl_min', default=0.2, type=float)
    parser.add_argument('--num_hidden_layers', default=2, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--checkpoint_frequency', default=10, type=int)
    parser.add_argument('--n_epochs', default=1000, type=int)
    parser.add_argument('--checkpoint_path', default="./", type=str)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--num_train', default=50000, type=int)
    parser.add_argument('--num_val', default=10000, type=int)
    parser.add_argument('--patience', default=5, type=int, help="Number of epochs metrics not improve")
    # parser.add_argument('--validate_every', default=4, type=int)
    parser.add_argument('--seed', default=0, type=int)

    args = parser.parse_args()
    return args
