import argparse
import random
import os
import utils
import torch
from torch.utils.tensorboard import SummaryWriter
from models import Model
from datasets import CoordDataset
from torch.utils.data import DataLoader

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

    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="batch size"
    )

    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="batch size"
    )
    parser.add_argument(
        "--lr",
        default=1e-3,
        type=float,
        help="optimizer lr"
    )

    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="number of epochs"
    )

    parser.add_argument(
        "--frequent",
        default=10,
        type=int,
        help="frequency of logging"
    )

    parser.add_argument(
        "--token_size",
        default=64,
        type=int,
        help="token size"
    )

    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-amp', dest='amp', action='store_false')
    parser.set_defaults(amp=False)

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

    img_resolution = (train_dataset.size[1], train_dataset.size[0])

    train_dataset = CoordDataset(train_dataset, img_resolution=img_resolution)
    val_dataset = CoordDataset(val_dataset, img_resolution=img_resolution)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = Model(args.token_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" : torch.backends.cudnn.benchmark = True

    
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    model = model.to(device)

    optim = torch.optim.Adam(lr=args.lr, params=model.parameters())

    loss = torch.nn.MSELoss().to(device)

    # logger.info(model)



    best_perf = 0.0
    best_model = False
    train_epochs = args.epochs

    for epoch in range(train_epochs):
        utils.train(train_dataloader, model, loss, optim, epoch, writer, logger, device, scaler, args)
        val_perf = utils.validate(val_dataloader, model, loss, epoch, writer, logger, device, img_resolution, args)

        if val_perf > best_perf:
            best_perf = val_perf
            best_model=True
        else:
            best_model = False
        logger.info("=> saving chekpoint to {}".format(logdir))
        utils.save_checkpoint({
            "epoch" : epoch + 1,
            "state_dict" : model.state_dict(),
            "perf" : val_perf,
            "last_epoch": epoch,
            "optimizer": optim.state_dict()
        }, best_model, logdir)
    
    final_model_state_file = os.path.join(logdir, "final_state.pth.tar")
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.state_dict(), final_model_state_file)
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    add_arguments(parser)
    args = parser.parse_args()
    main(args)