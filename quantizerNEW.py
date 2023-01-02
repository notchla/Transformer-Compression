import torch
import os 
import utils
import json
from models import Model
from datasets import CoordDataset
from timm.data.loader import MultiEpochsDataLoader
from utils import AverageMeter, lin2img, psnr_ssim, rescale_img
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


def flatten(compressed_list):
    """
    This function produces an array of json following the format {"fname":filename, "compressed", "quantized", "dequantized": lists of token_size numbers,}
    Compressed is the 32-bytes output of the architecture, quantized is a vector containing integer values in [0, 2^n_bits - 1].
    Dequantized is quantized rescaled in the range [-std_range*std + mean , std_range*std + mean].
    So, we can only store "quantized", reconstruct "dequantized" at runtime and pass it to the decoder.
    """
    ret = []
    for el in compressed_list:
        for i in range(len(el["fnames"])):
            compressed = el["original"][i]
            fname = el["fnames"][i]
            quantized = el["quantized"][i]
            dequantized = el["dequantized"][i]
            cmpr = {"fname" : fname, "compressed" : compressed.tolist(), "quantized":quantized.tolist(), "dequantized": dequantized.tolist()}
            ret.append(cmpr)
    return ret

def reload_trained_model(args):
    """
    Function to reload a pretrained model, following the naming scheme defined in this project.
    Returns the model, the dataset used for evaluation (ds), the dataloader, the device, the args used to train the model abd the image resolution
    """

    #reload arguments of trained model
    with open(os.path.join(args.model_path,"params.json"), "r") as f:
        parser_model = argparse.ArgumentParser()
        model_args = argparse.Namespace()
        d = json.load(f)

        #these parameters have been added in a later stage or were optional, so they are not present in some params.json files
        if "pos" not in d:
            d["pos"] = None
        if "amp" not in d:
            d["amp"] = None
        if "conditional" not in d:
            d["conditional"] = None
        if "custom_hidden_multiplier" not in d:
            d["custom_hidden_multiplier"] = None

        model_args.__dict__.update(d)

    train_dataset, val_dataset = utils.get_datasets(model_args)
    img_resolution = (train_dataset.img_resolution[1], train_dataset.img_resolution[0])

    ds = train_dataset if args.use_train_dataset else val_dataset 

    dataset = CoordDataset(ds, img_resolution=img_resolution)
    dataloader = MultiEpochsDataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = Model(model_args, img_resolution)
    device = torch.device("cpu")

    #reload weights from best model
    model.load_state_dict(torch.load(os.path.join(args.model_path, "model_best.pth.tar"), map_location=torch.device('cpu')))
        
    model.eval()

    return model, ds, dataloader, device, model_args, img_resolution


class Quantizer:
    """Class for quantizing image_codes. Takes as imput the mean and std of image_codes and the std_range.
       All values lying outside (-std_range, std_range) after normalization will be clipped to this range.
       This class is adapted from COIN++.
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        help="Path of the trained model",
        type=str,
        default="/cluster/home/fdeaglio/runs/42289", #TODO: change for submission
    )

    parser.add_argument(
        "--num_bits",
        help="List of number of bits at which to quantize ImageCodes.",
        nargs="+",
        type=int,
        default=[1,2, 4, 8, 16],
    )

    parser.add_argument(
        "--store",
        help="Subset of num_bits to be stored.",
        type=int,
        default=None,
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
    parser.add_argument(
        "--save_dir", 
        default='evaluations', 
        type=str, 
        help='path to save results')


    args = parser.parse_args()

    #get random number from trained model (typically ..../runs/12345 or ..../runs/12345/)
    run_id = args.model_path.split("/")[-1] if args.model_path[-1]!="/" else args.model_path.split("/")[-2]
    logdir = os.path.join(args.save_dir, str(run_id))

    model, ds, dataloader, device, model_args, img_resolution = reload_trained_model(args)


    #compute metrics on trained model
    loss = torch.nn.MSELoss().to(device)
    losses = AverageMeter()
    psnrs = AverageMeter()
    
    #compute metrics on reconstructed (and quantized) model
    losses_reconstructed = {}
    psnrs_reconstructed = {}
    for key in args.num_bits:
        losses_reconstructed[key] = AverageMeter()
        psnrs_reconstructed[key] = AverageMeter()

    #tensorboard for compressed images
    writer = SummaryWriter(logdir)

    compressed_list = []
    with torch.no_grad():
        #store intermediate results for later quantization. Needed since mean and stddev are computed once (and not once per batch)
        #For example, with a batch of 100 photos and a dataset of 1000, we can save only one vector of averages and only one of STDs instead of 10. 
        #This has a negligible impact on reconstruction but considerably reduces the space required.

        all_codes = [] #image codes
        all_gt_images = [] #groud truth images
        all_outputs = [] #outputs of the model (compressed images reconstructed from image code)
        all_coords = [] #coordinates 
        all_fnames = [] #file names

        for i, (input, target) in enumerate(dataloader):
            print(f"Compressing batch {i+1}/{len(dataloader)}")
            img = input["img"]
            coords = input["coords"]
            gt_image = target["img"]

            with torch.cuda.amp.autocast(enabled=model_args.amp):
                output = model(img, coords)
                image_codes = model.get_image_code(img, coords)
                loss_batch = loss(output, gt_image)

                all_codes.append(image_codes)
                all_outputs.append(output)
                all_coords.append(coords)
                all_fnames.append(ds.get_fnames(input["idx"]))

            num_images = img.size(0)
            losses.update(loss_batch.item(), num_images)
            psnr, ssim = psnr_ssim(lin2img(output, img_resolution), lin2img(gt_image, img_resolution))
            psnrs.update(psnr, num_images)

            all_gt_images.append(gt_image)
            

        #Evaluation of the non-quantized image-codes
        print("avg loss " + str(losses.avg))
        print("avg psnr " + str(psnrs.avg))

        all_together = torch.cat(all_codes) 
        #create the class to quantize
        quantizer = Quantizer(mean = all_together.mean(axis = 0), std = all_together.std(axis = 0), std_range=args.std_range)
        to_store = []

        for i in range(len(all_codes)):
            print(f"Quantizing batch {i+1}/100")          

            #retrieve previously stored data
            image_codes = all_codes[i]
            output = all_outputs[i]    
            gt_image = all_gt_images[i]
            coords = all_coords[i]
            fnames = all_fnames[i]
            
            #iterate over all required quantizations
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
                out_img_rec = rescale_img(out_img_rec, mode='clamp')
                compressed_all.append(out_img_rec)


                if args.store is not None and num_bits == args.store:
                    to_store.append({
                        "fnames": fnames,
                        "quantized":quantized,
                        "dequantized":dequantized,
                        "original": image_codes,
                        "num_bits":num_bits
                    })

            #not quantized version
            no_quantized = lin2img(output, img_resolution)  
            no_quantized = rescale_img(no_quantized, mode='clamp') 

            #ground truth
            gt_img = lin2img(gt_image, img_resolution)
            gt_img = rescale_img(gt_img, mode='clamp')

            #store on tensorboard the reconstructed images
            title = "_".join(["Original"]+[str(b)+"bits" for b in args.num_bits]+["noQuantized"])
            to_grid = torch.cat((gt_img, *compressed_all, no_quantized), dim=-1)
            writer.add_image(title, make_grid(to_grid), global_step=i)

        if args.store is not None:
            #"unbatch" the data 
            flattened = flatten(to_store)
            with open(os.path.join(logdir, f"quantized{str(args.store)}.json"), "w") as f:
                json.dump(flattened, f, ensure_ascii=False, indent=4)

        #print metrics for every quantization
        print("avg loss rec " + str([(l,losses_reconstructed[l].avg) for l in losses_reconstructed]))
        print("avg psnr rec " + str([(l, psnrs_reconstructed[l].avg) for l in psnrs_reconstructed]))
    
    
