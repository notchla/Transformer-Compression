# Deep Learning



## Download the datasets

Either run the script data_downloader.sh or use the following commands.

### CIFAR10
```
directory=/cluster/scratch/$user/datasets/

echo "Downloading data..."
wget https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz -P $directory
tar zxvf $directory/cifar10.tgz -C $directory
rm $directory/cifar10.tgz
```

### DIV2K
```
directory=/cluster/scratch/$user/datasets/DIV2K

echo "Downloading train..."
wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip -P $directory
unzip $directory/DIV2K_train_HR.zip -d $directory
rm  $directory/DIV2K_train_HR.zip

echo "Downloading valid..."
wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip -P $directory
unzip $directory/DIV2K_valid_HR.zip -d $directory
rm  $directory/DIV2K_valid_HR.zip

```

## Train the model

Here the commands to train the model from scratch. Training could require few hours, so we provide a pretrained model as well.

Our suggested combination is 
```
python main.py --token_size 256 --lr 0.0005 --train_dataset CIFAR10 --val_dataset CIFAR10 --data_root <REPLACE_WITH_YOUR_DIRECTORY> --frequent 150 --custom_hidden_multiplier 2
```

Command explaination and further options can be found in the help
```
python main.py --help
```
## Evaluate

Results are provided diring training but you can use a standalone script to evaluate the model (PSNR) also using the compression. 

For this part, we provided our pretrained model that can be found in 
```
pretrained/
```

The script for evaluation has several options: you can specify the dataset (--evaluation_dataset, along with the path where data is stored --data_root), the number of bits and so forth.

Compression and quantization work only on CPU, please allow longer compressing times.

### Compressing CIFAR

CIFAR10 is the default dataset, so you can just launch
```
python quantizerBenchmark.py --data_root <REPLACE_WITH_YOUR_DIRECTORY> 
```
It produces a tensorboard with the reconstructed images in the folder
```
evaluations/run_id
```
Where run_id is a random number printed at the beginning of execution. The script prints the mean for the not quantized and for each quantized version
### Compressing KODAK and DIV2K

These datasets are a little bit trickier since images need to be patched. We suggest to use a batch_size of 384 (so each batch is an image) and to compress one image at a time (using --image_index).

```
python quantizerBenchmark.py --evaluation_dataset DIV2K --image_index 38 --data_root <REPLACE_WITH_YOUR_DIRECTORY> --batch_size 384
```

You can find a tensorboard with the reconstructed image also for these datasets.
