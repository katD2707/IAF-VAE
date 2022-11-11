import torch
import torch.optim as optim
from argparse import ArgumentParser
import yaml
from torchvision import transforms
import utils, datasets, models
import numpy as np
import os
from tensorboardX import SummaryWriter


def train(params):
    # Set random seed for reproducibility
    utils.set_seed(params['generic']['seed'])

    # Get GPU if present
    device = utils.get_device()

    # Set number of threads
    if params['generic']['workers'] > 0:
        torch.set_num_threads(params['generic']['workers'])

    # Get transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x - 0.5,
    ])

    # Get data
    dataset = datasets.Cifar10(train=True,
                               val=True,
                               root=params['dataset']['data_path'],
                               transform_train=transform,
                               transform_val=transform,
                               download=True,
                               )
    # Get data loader
    train_loader, val_loader = dataset.get_dataloader(params['dataset']['batch_size'],
                                                      num_workers=params['dataset']['num_workers'],
                                                      pin_memory=True,
                                                      )

    # Get model
    model = models.CVAE(in_channels=params['model']['in_channels'],
                        hidden_size=params['model']['hidden_size'],
                        z_size=params['model']['z_size'],
                        batch_size=params['dataset']['batch_size'],
                        k=params['model']['k'],
                        kl_min=params['model']['kl_min'],
                        num_hidden_layers=params['model']['num_hidden_layers'],
                        num_blocks=params['model']['num_blocks'],
                        image_size=params['dataset']['image_size'],
                        )
    model.to(device)

    # Optimizer
    optimizer = optim.Adamax(model.parameters(), lr=params['training']['optimizer']['learning_rate'])

    # spawn writer
    model_name = 'NB{}_D{}_Z{}_H{}_BS{}_LR{}'.format(params['model']['num_blocks'],
                                                     params['model']['num_hidden_layers'], params['model']['z_size'],
                                                     params['model']['hidden_size'],
                                                     params['dataset']['batch_size'],
                                                     params['training']['optimizer']['learning_rate'])

    log_dir = os.path.join('runs', model_name)
    sample_dir = os.path.join(log_dir, 'samples')
    writer = SummaryWriter(log_dir=log_dir)
    utils.maybe_create_dir(sample_dir)

    utils.print_and_save_args(args, log_dir)
    print('logging into %s' % log_dir)
    utils.maybe_create_dir(sample_dir)
    best_test = float('inf')

    print('Start training...')
    for epoch in range(start_epoch, params['training']['n_epochs']):
        model.train()
        losses = 0.
        avg_bpd = 0.
        train_log = utils.reset_log()

        for batch_idx, (inputs, _) in enumerate(train_loader):
            inputs = inputs.cuda()
            x, loss, elbo = model(inputs)

            loss = loss / x.shape[0]
            losses += loss
            bpd = elbo / (params['dataset']['image_size'] ** 2 * params['model']['in_channels'] * np.log(2.))
            avg_bpd += bpd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_log['bpd'] += [bpd]
            train_log['elbo'] += [elbo]

            if batch_idx % 25 == 0:
                print(f'Epoch: {epoch + 1} | Step: {batch_idx + 1}/{len(train_loader)} | Loss: {loss} |'
                      f'Bits/Dim: {bpd/batch_idx+1}')

        print(f'Epoch: {epoch + 1} | Step: {batch_idx + 1}/{len(train_loader)} | Loss: {losses/len(train_loader)} |'
              f'Bits/Dim: {avg_bpd/len(train_loader)}')

        loss = 0.
        bpd = 0.
        elbo = 0.
        step = 0
        model.eval()
        for batch_idx, (inputs, _) in enumerate(val_loader):
            inputs = inputs.cuda()
            x, obj, loss = model(inputs)
            step += 1
            obj = obj / x.shape[0]
            bpd = loss / (params.dataset.image_size ** 2 * 3 * np.log(2.))


if __name__ == "__main__":
    # Parse parameters
    parser = ArgumentParser(description="train ResnetVAE model")
    parser.add_argument(
        "-p",
        "--config_path",
        help="path to the config file",
        required=True,
        type=str,
    )
    args = parser.parse_args()
    with open(args.config_path, "r") as params:
        args = yaml.load(params, Loader=yaml.FullLoader)

    print(args)
    train(args)
