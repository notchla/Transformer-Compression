import logging
import datetime
import os
import json
import datasets

def get_logger(logdir):
    logger = logging.getLogger('emotion')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger

def save_config(logdir, config):
    param_path = os.path.join(logdir, "params.json")
    print("[*] PARAM path: %s" % param_path)
    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def get_datasets(args):
    if "DIV2K" in (args.train_dataset, args.val_dataset):
        if args.train_dataset == "DIV2K":
            train_dataset = datasets.DIV2K("train", args.data_root)
        if args.val_dataset == "DIV2K":
            val_dataset = datasets.DIV2K("val", args.data_root)
    return train_dataset, val_dataset