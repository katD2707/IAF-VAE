import torch
import torch.optim as optim
from argparse import ArgumentParser
import yaml
from torchvision import transforms
import utils, datasets, models
import numpy as np
import os
from tensorboardX import SummaryWriter
from torchvision.utils import save_image


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
                        device=device,
                        )

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

    start_epoch = 0
    if params.current_checkpoint is not None:
        if os.path.exists(params.current_checkpoint) is True:
            checkpoint = torch.load(params.current_checkpoint)
            model.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint['epoch']
    model.to(device)

    print('Start training...')
    for epoch in range(start_epoch, params['training']['n_epochs']):
        model.train()
        losses = []
        avg_bpd = []
        train_log = utils.reset_log()

        for batch_idx, (inputs, _) in enumerate(train_loader):
            inputs = inputs.cuda()
            x, loss, elbo = model(inputs)

            loss = loss / x.shape[0]
            losses.append(loss)
            bpd = elbo / (params['dataset']['image_size'] ** 2 * params['model']['in_channels'] * np.log(2.))
            avg_bpd.append(bpd.mean())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_log['bpd'] += [bpd.mean()]
            train_log['elbo'] += [elbo.mean()]

            if batch_idx % 25 == 0:
                print(f'Epoch: {epoch + 1} | Step: {batch_idx + 1}/{len(train_loader)} | Loss: {loss:.2f} | '
                      f'Bits/Dim: {bpd.mean():.2f}')

        for key, value in train_log.items():
            utils.print_and_log_scalar(writer, 'train/%s' % key, value, epoch)

        print(f'===> Epoch: {epoch + 1} | Loss: {sum(losses) / len(losses):.2f} | '
              f'Bits/Dim: {sum(avg_bpd) / len(avg_bpd):.2f}')

        model.eval()
        losses = []
        avg_bpd = []
        test_log = utils.reset_log()

        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(val_loader):
                inputs = inputs.cuda()
                x, loss, elbo = model(inputs)

                loss = loss / x.shape[0]
                losses.append(loss)
                bpd = elbo / (params['dataset']['image_size'] ** 2 * params['model']['in_channels'] * np.log(2.))
                avg_bpd.append(bpd.mean())

                test_log['bpd'] += [bpd.mean()]
                test_log['elbo'] += [elbo.mean()]

                all_samples = model.cond_sample(inputs)
                # save reconstructions
                out = torch.stack((x, inputs))  # 2, bs, 3, 32, 32
                out = out.transpose(1, 0).contiguous()  # bs, 2, 3, 32, 32
                out = out.view(-1, x.size(-3), x.size(-2), x.size(-1))

                all_samples += [x]
                all_samples = torch.stack(all_samples)  # L, bs, 3, 32, 32
                all_samples = all_samples.transpose(1, 0)
                all_samples = all_samples.contiguous()  # bs, L, 3, 32, 32
                all_samples = all_samples.view(-1, x.size(-3), x.size(-2), x.size(-1))

                save_image(utils.scale_inv(all_samples), os.path.join(sample_dir, 'test_levels_{}.png'.format(epoch)),
                           nrow=12)
                save_image(utils.scale_inv(out), os.path.join(sample_dir, 'test_recon_{}.png'.format(epoch)), nrow=12)

            save_image(utils.scale_inv(model.sample(64)), os.path.join(sample_dir, 'sample_{}.png'.format(epoch)),
                       nrow=8)

            print(f'===> Validation | Epoch: {epoch + 1} | Loss: {sum(losses) / len(losses):.2f} | '
                  f'Bits/Dim: {sum(avg_bpd) / len(avg_bpd):.2f}')

            for key, value in test_log.items():
                utils.print_and_log_scalar(writer, 'test/%s' % key, value, epoch)

        if (epoch + 1) % params['training']['checkpoints_frequency'] == 0 or epoch == 0:
            # Save model checkpoint and epoch
            state_dict = {
                "model": model.state_dict(),
                "epoch": epoch,
            }
            torch.save(state_dict, os.path.join(log_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

        current_test = sum(test_log['bpd']) / len(val_loader)
        if current_test < best_test:
            best_test = current_test
            print('saving best model')
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))


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
    parser.add_argument(
        "-ckpt",
        "--current_checkpoint",
        default=None,
        type=str,
    )
    args = parser.parse_args()
    with open(args.config_path, "r") as params:
        args = yaml.load(params, Loader=yaml.FullLoader)

    train(args)
