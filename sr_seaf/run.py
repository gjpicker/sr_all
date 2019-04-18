import torch 
import dataset as dt 
import trainer as tr 
import torch.utils.data as t_data



if __name__=="__main__":
    x_dir ="/home/wangjian7/download/fds_data/data/data/SR_data/"
    dt_c = dt. TrainDatasetFromFolder(image_dirs=x_dir)


    class config :
        lr =0.0001
        betas =( 0.9 ,0.99)
        lambda_r= 0.1
        epoches =10

    kw ={"pin_memory":True , "num_workers":8} if torch.cuda.is_available() else {} 
    dl_c =t_data.DataLoader(dt_c ,batch_size=1, **kw)

    tr_l = tr.Treainer(opt=config() ,train_dt = dl_c )
    tr_l.run()

