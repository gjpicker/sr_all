import argparse
import json 

import torch 
import dataset as dt 
import trainer as tr 
import torch.utils.data as t_data

from collections import namedtuple

#=====START: ADDED FOR DISTRIBUTED======
#from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
#from torch.utils.data.distributed import DistributedSampler
#=====END:   ADDED FOR DISTRIBUTED======

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

num_gpus = torch.cuda.device_count()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    train_config = namedtuple("config", train_config.keys())(*train_config.values() )
    #global data_config
    #data_config = config["data_config"]
    dist_config = config["dist_config"]
    #global waveglow_config
    #waveglow_config = config["waveglow_config"]

    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set, detect %d"%(num_gpus))
            print("Only running 1 GPU.  Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")



    x_dir_vid2k ="/home/wangjian7/download/fds_data/data/data/SR_data_index.log"
    x_dir_imagenet = "/home/wangjian7/download/fds_data/data/data/Imagenet_400x400_data_index.log"#"/home/wangjian7/download/fds_data/data/data/ILSVRC2017_data_index.log"


    #=====START: ADDED FOR DISTRIBUTED======
    #if num_gpus > 1:
    #    init_distributed(args.rank, num_gpus, args.group_name, **dist_config)
    #=====END:   ADDED FOR DISTRIBUTED======


    dt_gan = dt. TrainDatasetFromFolder(image_index_path=x_dir_vid2k)
    dt_pre = dt. TrainDatasetFromFolder(image_index_path=x_dir_imagenet)
    # =====START: ADDED FOR DISTRIBUTED======
    #train_sampler = DistributedSampler(dt_c) if num_gpus > 1 else None
    # =====END:   ADDED FOR DISTRIBUTED======

#    kw ={"pin_memory":True , "num_workers":8} if torch.cuda.is_available() else {} 
#    dl_c =t_data.DataLoader(dt_c ,batch_size=1, **kw , sampler=train_sampler , drop_last=True)
    print (type(train_config) ,train_config)

    dist_info = [args.rank, num_gpus, args.group_name, dist_config]
    tr_l = tr.Treainer(opt=train_config  ,\
        train_dt_warm= dt_pre,
        train_dt = dt_gan , dis_list =  dist_info)
    tr_l.run()

