#old, to be cancelled


import torch
import os 
import utils
from models import Model
from datasets import CoordDataset
from timm.data.loader import MultiEpochsDataLoader
from utils import AverageMeter, lin2img, psnr_ssim, rescale_img
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

def reload_trained_model(args):
    #get random number from trained model (typically ..../runs/12345 or ..../runs/12345/)
    run_id = args.model_path.split("/")[-1] if args.model_path[-1]!="/" else args.model_path.split("/")[-2]

    #reload arguments of trained model
    with open(os.path.join(args.model_path,"params.json"), "r") as f:
        parser_model = argparse.ArgumentParser()
        model_args = argparse.Namespace()
        d = json.load(f)
        if "pos" not in d:
            d["pos"] = None
        if "amp" not in d:
            d["amp"] = None
        if "conditional" not in d:
            d["conditional"] = None
        if "custom_hidden_multiplier" not in d:
            d["custom_hidden_multiplier"] = None
        model_args.__dict__.update(d)
        print(d)

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
        model.load_state_dict(torch.load(os.path.join(args.model_path,"model_best.pth.tar")))
    else:
        model.load_state_dict(torch.load(os.path.join(args.model_path, "model_best.pth.tar"), map_location=torch.device('cpu')))
        
    model.eval()

    return model, ds, dataloader, device, model_args, img_resolution


class Quantizer:
    """Class for quantizing image_codes
    Args:
        mean (float): Mean of image_codes.
        std (float): Standard deviation of image_codes.
        std_range (float): Number of standard deviations defining quantization
            range. All values lying outside (-std_range, std_range) after
            normalization will be clipped to this range.
    """

    def __init__(self, mean, std, std_range=3.0):
        self.mean = mean
        self.std = std
        self.std_range = std_range

    def quantize(self, image_codes, num_bits):
        """Uniformly quantize image_codes to a given number of bits.
        Args:
            image_codes (torch.Tensor):
            num_bits (int): Number of bits at which to quantize. This
                corresponds to uniformly quantizing into 2 ** num_bits bins.
        """
        # Normalize image_codes
        norm_codes = (image_codes - self.mean) / self.std
        # Clip image_codes to lie in quantization range
        norm_codes = torch.clamp(norm_codes, -self.std_range, self.std_range)
        # Map image_codes from [-std_range, std_range] to [0, 1]
        norm_codes = norm_codes / (2 * self.std_range) + 0.5
        # Compute number of bins
        num_bins = 2**num_bits
        # Quantize image_codes. After multiplying by (num_bins - 1) this will
        # yield values in [0, num_bins - 1]. Rounding will then yield values in
        # {0, 1, ..., num_bins - 1}, i.e. num_bins different values
        quantized_codes = torch.round(norm_codes * (num_bins - 1))
        # Dequantize image_codes
        dequantized_norm_codes = quantized_codes / (num_bins - 1)
        dequantized_norm_codes = (dequantized_norm_codes - 0.5) * 2 * self.std_range
        dequantized_codes = dequantized_norm_codes * self.std + self.mean
        return quantized_codes.int(), dequantized_codes


import json

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        help="Path of the trained model",
        type=str,
        default="/cluster/home/fdeaglio/runs/42289",
    )

    parser.add_argument(
        "--num_bits",
        help="List of number of bits at which to quantize ImageCodes.",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16],
    )

    parser.add_argument(
        "--batch_size",
        help="Batch size to use when evaluating ImageCodes.",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--std_range",
        help="Quantization range (in number of standard devs).",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--use_train_dataset",
        help="Whether to get image_codes of train dataset. If False, get of test set.",
        type=int,
        default=0,
    )
    parser.add_argument("--save_dir", default='evaluations', type=str, help='path to save results')


    args = parser.parse_args()

    #get random number from trained model (typically ..../runs/12345 or ..../runs/12345/)
    run_id = args.model_path.split("/")[-1] if args.model_path[-1]!="/" else args.model_path.split("/")[-2]
    logdir = os.path.join(args.save_dir, str(run_id))

    model, ds, dataloader, device, model_args, img_resolution = reload_trained_model(args)


    #compute metrics on trained model
    loss = torch.nn.MSELoss().to(device)
    losses = AverageMeter()
    psnrs = AverageMeter()
    
    losses_reconstructed = {}
    psnrs_reconstructed = {}
    for key in args.num_bits:
        losses_reconstructed[key] = AverageMeter()
        psnrs_reconstructed[key] = AverageMeter()

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
            
            #create the class to quantize
            quantizer = Quantizer(mean = image_codes.mean(axis = 0), std = image_codes.std(axis = 0), std_range=args.std_range)
            
            compressed_all = []
            for num_bits in args.num_bits:
                #quantize the weights
                quantized, dequantized = quantizer.quantize(image_codes, num_bits=num_bits)
                #get the reconstructed images
                with torch.cuda.amp.autocast(enabled=model_args.amp):
                    reconstructed_output = model.reconstruct(dequantized, coords)
                    loss_batch_reconstructed = loss(reconstructed_output, gt_image)
                
                num_images = img.size(0)
                losses_reconstructed[num_bits].update(loss_batch_reconstructed.item(), num_images)
                psnr, ssim = psnr_ssim(lin2img(reconstructed_output, img_resolution), lin2img(gt_image, img_resolution))
                psnrs_reconstructed[num_bits].update(psnr, num_images)

                out_img_rec = lin2img(reconstructed_output, img_resolution)   
                out_img_rec = rescale_img(out_img, mode='clamp')
                compressed_all.append(out_img_rec)

                output_vs_gt_vs_rec = torch.cat((out_img, gt_img, out_img_rec), dim=-1)
                writer.add_image("Batch"+ str(i) + f"_pred_vs_gt_vs_rec", make_grid(output_vs_gt_vs_rec), global_step=num_bits)
            title = "_".join(["gt"]+[str(b) for b in args.num_bits])
            to_grid = torch.cat((gt_img, *compressed_all), dim=-1)
            writer.add_image(title, make_grid(to_grid), global_step=i)

            print("avg loss " + str(losses.avg))
            print("avg psnr " + str(psnrs.avg))
            print("avg loss rec " + str([(l,losses_reconstructed[l].avg) for l in losses_reconstructed]))
            print("avg psnr rec " + str([(l, psnrs_reconstructed[l].avg) for l in psnrs_reconstructed]))
    
    
