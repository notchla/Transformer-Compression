#!/bin/bash 
 
 
 
user=$(whoami) 
 
echo "DIV2K" 
directory=/cluster/scratch/$user/datasets/DIV2K 
 
echo "Downloading train..." 
wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip -P $directory 
unzip $directory/DIV2K_train_HR.zip -d $directory 
rm  $directory/DIV2K_train_HR.zip 
 
echo "Downloading valid..." 
wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip -P $directory 
unzip $directory/DIV2K_valid_HR.zip -d $directory 
rm  $directory/DIV2K_valid_HR.zip 
 
 
echo "cifar10" 
directory=/cluster/scratch/$user/datasets/ 
 
echo "Downloading data..." 
wget https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz -P $directory 
tar zxvf $directory/cifar10.tgz -C $directory 
rm $directory/cifar10.tgz
