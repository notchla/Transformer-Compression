import argparse
import random
import os
import utils
import torch
from torch.utils.tensorboard import SummaryWriter
from models import Encoder, Decoder


def add_arguments(parser):
    parser.add_argument("--save_dir", default='runs', type=str, help='path to save trained models and logs')
    parser.add_argument("--data_root", default='datasets', type=str, help='path to dataset folder')

    parser.add_argument(
        "--train_dataset",
        default="DIV2K",
        choices=(
            "DIV2K"
        ),
    )

    parser.add_argument(
        "--val_dataset",
        default="DIV2K",
        choices=(
            "DIV2K"
        ),
)

def main(args):
    print(args)
    run_id = random.randint(1, 100000)
    logdir = os.path.join(args.save_dir, str(run_id))
    writer = SummaryWriter(logdir)
    logger = utils.get_logger(logdir)
    print("LOGDIR : {}".format(logdir))
    utils.save_config(logdir, args)
    logger.info("start")

    train_dataset, val_dataset = utils.get_datasets(args)

    logger.info((len(train_dataset), len(val_dataset)))

    x = torch.rand((5, 224*224, 64))

    logger.info(x.shape)

    encoder = Encoder(embed_dim=64)

    logger.info(encoder)

    cls = encoder(x)

    logger.info(cls.shape)

    decoder = Decoder(embed_dim=64)

    logger.info(decoder)

    x = decoder(x, cls)

    logger.info(x.shape)




    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    add_arguments(parser)
    args = parser.parse_args()
    main(args)