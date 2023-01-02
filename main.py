import argparse
import random
import os
import json
import utils
import torch
from torch.utils.tensorboard import SummaryWriter
from models import Model
from datasets import CoordDataset
from torch.utils.data import DataLoader
from utils import get_mape_loss, PrefetchLoader
from timm.data.loader import MultiEpochsDataLoader

def reload_arguments(path):
    #reload arguments from an already trained model
    with open(os.path.join(path,"params.json"), "r") as f:
        parser_model = argparse.ArgumentParser()
        model_args = argparse.Namespace()
        d = json.load(f)
        if "pos" not in d:
            d["pos"] = None
        if "amp" not in d:
            d["amp"] = None
        if "conditional" not in d:
            d["conditional"] = None
        model_args.__dict__.update(d)
    return model_args

def add_arguments(parser):
    parser.add_argument("--save_dir", default='runs', type=str, help='path to save trained models and logs')
    parser.add_argument("--data_root", default='datasets', type=str, help='path to dataset folder')

    parser.add_argument(
        "--train_dataset",
        default="DIV2K",
        choices=(
            "DIV2K","CIFAR10",
        ),
    )

    parser.add_argument(
        "--val_dataset",
        default="DIV2K",
        choices=(
            "DIV2K","CIFAR10",
        ),
    )

    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="batch size"
    )

    parser.add_argument(
        "--num_workers",
        default=16,
        type=int,
        help="batch size"
    )
    parser.add_argument(
        "--lr",
        default=1e-5,
        type=float,
        help="optimizer lr"
    )

    parser.add_argument(
        "--epochs",
        default=1600,
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
        "--val_frequency",
        default=10,
        type=int,
        help="number of epochs between validation"
    )

    parser.add_argument(
        "--token_size",
        default=64,
        type=int,
        help="token size"
    )

    parser.add_argument(
        "--patch_size",
        default=32,
        type=int,
        help="random crop size"
    )

    parser.add_argument(
        "--ff_dims",
        default=16,
        type=int,
        help="ff_dims for positional encoding"
    )

    parser.add_argument(
        "--encoding_scale",
        default=1.4,
        type=int,
        help="encoding scale for positional encoding"
    )

    parser.add_argument(
        "--custom_hidden_multiplier",
        default=None,
        type=int,
        help="If None, the hidden layers of MLPs have the dimension of token_size/2. Otherwise, they can be set to custom_hidden_multiplier*token_size/2"
    )

    parser.add_argument(
        "--conditional",
        nargs='*',
        type=str,
        help="train only on a subset of classes (CIFAR10) Classes: airplane  automobile  bird  cat  deer  dog  frog  horse  ship  truck"
    )

    parser.add_argument(
        "--restart_training",
        default=False,
        help="restart from a given checkpoint. Specify the full path of the folder. Loads model_best.pth.tar and params.json. Other args are ignored."
    )

    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-amp', dest='amp', action='store_false')
    parser.set_defaults(amp=False)

    parser.add_argument('--pos', action='store_true')
    parser.add_argument('--no-pos', dest='pos', action='store_false')
    parser.set_defaults(pos=True)

    parser.add_argument('--prefetcher', action='store_true')
    parser.add_argument('--no-prefetcher', dest='prefetcher', action='store_false')
    parser.set_defaults(prefetcher=True)

def main(args):
    print(args)
    reload_path = None
    if args.restart_training:
        reload_path = args.restart_training
        args = reload_arguments(reload_path)
        print("Reloaded args: ",args)

    run_id = random.randint(1, 100000)
    logdir = os.path.join(args.save_dir, str(run_id))
    writer = SummaryWriter(logdir)
    logger = utils.get_logger(logdir)
    print("LOGDIR : {}".format(logdir))
    utils.save_config(logdir, args)
    logger.info("start")

    train_dataset, val_dataset = utils.get_datasets(args)

    logger.info((len(train_dataset), len(val_dataset)))

    img_resolution = (train_dataset.img_resolution[1], train_dataset.img_resolution[0])

    train_dataset = CoordDataset(train_dataset, img_resolution=img_resolution)
    val_dataset = CoordDataset(val_dataset, img_resolution=img_resolution)

    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    train_dataloader = MultiEpochsDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = MultiEpochsDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.prefetcher:
        train_dataloader = PrefetchLoader(train_dataloader)
        val_dataloader = PrefetchLoader(val_dataloader)

    model = Model(args, img_resolution)

    if reload_path is not None:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(os.path.join(reload_path,"model_best.pth.tar")))
        else:
            model.load_state_dict(torch.load(os.path.join(reload_path, "model_best.pth.tar"), map_location=torch.device('cpu')))
        logger.info(f"Reloaded weights from run: {reload_path}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" : torch.backends.cudnn.benchmark = True

    
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    model = model.to(device)

    optim = torch.optim.Adam(lr=args.lr, params=model.parameters())

    # loss = get_mape_loss()
    #loss = torch.nn.L1Loss().to(device)
    
    loss = torch.nn.MSELoss().to(device)

    logger.info(model)



    best_perf = 0.0
    best_model = False
    train_epochs = args.epochs

    for epoch in range(train_epochs):
        utils.train(train_dataloader, model, loss, optim, epoch, writer, logger, device, scaler, args)
        if epoch % args.val_frequency == 0:
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
    logger.info("best model val psnr {}".format(best_perf))
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    add_arguments(parser)
    args = parser.parse_args()
    main(args)