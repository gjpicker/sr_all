import argparse
# import json 
import yaml 

import torch 
import dataset as dt 
import trainer as tr 
import torch.utils.data as t_data

from collections import namedtuple
import os 
#=====START: ADDED FOR DISTRIBUTED======
#from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
#from torch.utils.data.distributed import DistributedSampler
#=====END:   ADDED FOR DISTRIBUTED======

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

num_gpus = torch.cuda.device_count()
# 
# def metatuple(name, attrs):
#     '''
#     https://gist.github.com/caruccio/3765157
#     '''
#     class _meta_cls(type):
#         '''Metaclass to replace namedtuple().__new__ with a recursive version _meta_new().'''
#         def __new__(mcs, name, bases, metadict):
#             def _meta_new(_cls, **kv):
#                 return tuple.__new__(_cls, ([ (metatuple('%s_%s' % (_cls.__name__, k), kv[k].keys())(**kv[k]) if isinstance(kv[k], dict) else kv[k]) for k in _cls._fields]))
#             metadict['__new__'] = _meta_new
#             return type.__new__(mcs, bases[0].__name__, bases, metadict)
# 
#     class _metabase(namedtuple(name, ' '.join(attrs))):
#         '''Wrapper metaclass for namedtuple'''
#         __metaclass__ = _meta_cls
# 
#     return _metabase

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='YAML file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
#     with open(args.config) as f:
#         data = f.read()
#     config = json.loads(data)
    config = yaml.load(open(args.config))
    train_config = config["train_config"]
    ###########contraint
    train_config["checkpoints"]="save_ck"
    train_config["name"]= os.path.basename(str(args.config)).replace(".","_")
    

    
    train_config["dis"] = namedtuple("dis", train_config["dis"].keys())(*train_config["dis"].values() )
    train_config["gen"] = namedtuple("gen", train_config["gen"].keys())(*train_config["gen"].values() )
    train_config["warm_opt"] = namedtuple("warm_opt", train_config["warm_opt"].keys())(*train_config["warm_opt"].values() )
    train_config = namedtuple("config", train_config.keys())(*train_config.values() )
#     train_config = metatuple("config", train_config)
#     print (dir(train_config),dir(train_config.dis),"---"*10)
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



    x_dir_vid2k =train_config.vid2k_data_root
    x_dir_imagenet = train_config.imagenet_data_root 


    if config["is_debug"]:
        is_debug_size=train_config.batch_size
        dt_pre = dt. TrainDatasetFromFolder(image_index_path=x_dir_imagenet,is_debug_size=is_debug_size)
        dt_gan = dt. TrainDatasetFromFolder(image_index_path=x_dir_vid2k,is_debug_size=is_debug_size)
        print ("debug mode::size->",len(dt_gan))
        dt_gan_val = dt. TestDatasetFromFolder(image_index_path=x_dir_vid2k,is_debug_size=is_debug_size)
    else :
        dt_pre = dt. TrainDatasetFromFolder(image_index_path=x_dir_imagenet)
        dt_gan = dt. TrainDatasetFromFolder(image_index_path=x_dir_vid2k)
        dt_gan_val = dt. TestDatasetFromFolder(image_index_path=x_dir_vid2k)
    # =====START: ADDED FOR DISTRIBUTED======
    #train_sampler = DistributedSampler(dt_c) if num_gpus > 1 else None
    # =====END:   ADDED FOR DISTRIBUTED======

#    kw ={"pin_memory":True , "num_workers":8} if torch.cuda.is_available() else {} 
#    dl_c =t_data.DataLoader(dt_c ,batch_size=1, **kw , sampler=train_sampler , drop_last=True)
    print (type(train_config) ,train_config)


    dist_info = [args.rank, num_gpus, args.group_name, dist_config]
    tr_l = tr.Treainer(opt=train_config  ,\
        train_dt_warm= dt_pre,
        train_dt = dt_gan , dis_list =  dist_info, val_dt_warm = dt_gan_val )
    tr_l.run()

