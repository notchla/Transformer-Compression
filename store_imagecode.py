#old, to be cancelled

import time
import torch
from models import Model
import utils
import os, json
from datasets import CoordDataset
from timm.data.loader import MultiEpochsDataLoader
from utils import AverageMeter, lin2img, psnr_ssim, rescale_img
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import json

def flatten(compressed_list):
    """
    This function produces an array of json following the format {"fname":filename, "compressed": list of token_size numbers}
    """
    ret = []
    for el in compressed_list:
        for i in range(len(el["files"])):
            compressed = el["image_codes"][i]
            fname = el["files"][i]
            cmpr = {"fname" : fname, "compressed" : compressed.tolist()}
            ret.append(cmpr)
    return ret

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        help="path of a trained model.",
        type=str,
        default=os.path.join(os.pardir, "runs", "84217"),
    )

    parser.add_argument("--save_dir", default='evaluations', type=str, help='path to save results')

    parser.add_argument(
        "--use_train_dataset",
        help="Whether to get image_codes of train dataset. If False, get of test set.",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--batch_size",
        help="Batch size to use when getting image codes.",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--store",
        help="Whether to store image codes.",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    #get random number from trained model (typically ..../runs/12345 or ..../runs/12345/)
    run_id = args.path.split("/")[-1] if args.path[-1]!="/" else args.path.split("/")[-2]
    logdir = os.path.join(args.save_dir, str(run_id))


    #reload arguments of trained model
    with open(os.path.join(args.path,"params.json"), "r") as f:
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
    
    train_dataset, val_dataset = utils.get_datasets(model_args)
    img_resolution = (train_dataset.img_resolution[1], train_dataset.img_resolution[0])

    if args.use_train_dataset:
        ds = train_dataset
    else:
        ds = val_dataset
    
    dataset = CoordDataset(ds, img_resolution=img_resolution)
    dataloader = MultiEpochsDataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = Model(model_args, img_resolution)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" : torch.backends.cudnn.benchmark = True


    #reload weights from best model
    if device == "cuda":
        model.load_state_dict(torch.load(os.path.join(args.path,"model_best.pth.tar")))
    else:
        model.load_state_dict(torch.load(os.path.join(args.path, "model_best.pth.tar"), map_location=torch.device('cpu')))
        
    model.eval()

    #compute metrics on trained model
    loss = torch.nn.MSELoss().to(device)
    losses = AverageMeter()
    psnrs = AverageMeter()

    #tensorboard for compressed images
    writer = SummaryWriter(logdir)

    compressed_list = []
    with torch.no_grad():
        for i, (input, target) in enumerate(dataloader):
            print(f"Compressing batch {i+1}/{len(dataloader)}")
            img = input["img"]
            coords = input["coords"]
            gt_image = target["img"]

            img = img.to(device)
            coords = coords.to(device)
            gt_image = gt_image.to(device)

            with torch.cuda.amp.autocast(enabled=model_args.amp):
                output = model(img, coords)
                image_codes = model.get_image_code(img, coords)
                loss_batch = loss(output, gt_image)

            num_images = img.size(0)
            losses.update(loss_batch.item(), num_images)
            psnr, ssim = psnr_ssim(lin2img(output, img_resolution), lin2img(gt_image, img_resolution))
            psnrs.update(psnr, num_images)

            out_img = lin2img(output, img_resolution)   
            gt_img = lin2img(gt_image, img_resolution)
            out_img = rescale_img(out_img, mode='clamp')
            gt_img = rescale_img(gt_img, mode='clamp')


            output_vs_gt = torch.cat((out_img, gt_img), dim=-1)
            writer.add_image("Batch"+ str(i) + "_pred_vs_gt", make_grid(output_vs_gt), global_step=i)
            
            compressed = {"image_codes": image_codes.cpu().numpy(), "files": ds.get_fnames(input["idx"])}
            
            compressed_list.append(compressed)

        flattened = flatten(compressed_list)
        if args.store:
            with open(os.path.join(logdir, "compressed.json"), "w") as f:
                print(json.dumps(flattened), file = f)
        print("avg loss " + str(losses.avg))
        print("avg psnr " + str(psnrs.avg))
    
    
