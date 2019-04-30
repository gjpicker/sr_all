import torch
from models.networks import *
if __name__ == "__main__":
    import  unittest
    import yaml 
    from collections import namedtuple

    class TestAll(unittest.TestCase):

        def getconfig(self,strx):
            train_config = yaml.load(strx)["train_conf"]
            train_config["dis"] = namedtuple("dis", train_config["dis"].keys())(*train_config["dis"].values() )
            train_config["gen"] = namedtuple("gen", train_config["gen"].keys())(*train_config["gen"].values() )
            train_config = namedtuple("config", train_config.keys())(*train_config.values() )
            return train_config
        def setUp(self):
            strx = """
train_conf :
    vgg : transfer
    model_dir : ./
    dis :
        net_type: pixel #nlayer ,pixel ,multi 
        num_d : 3 # if net_type==multi ,return cascad range(D)
    gen :
        net_type : carnm #carnm carn carn_gan carn_ganm
            """
            
            self.opt =self.getconfig(strx) 
            if self.opt . vgg =="transfer":
                m = Vgg16 ()
                if not os.path.isfile(os.path.join(self.opt.model_dir ,"vgg16.weight")):
                    torch.save(m.state_dict(),os.path.join(self.opt.model_dir,"vgg16.weight")  )


#         def test_vgg(self):
#             print (self.opt)
#             ##mock
#             vgg = define_vgg(self.opt )
#             self.assertEqual(len([x for x in vgg.named_children()]) ,13 )
# 
#             setattr(self.opt ,"vgg" ,"classify")
#             vgg = define_vgg(self.opt )
#             self.assertEqual(len([x for x in vgg.named_children()]) ,1 )

        def getnetG_list(self):
            x1={}
            
            x1.update({"carn": define_G(ge_net_str="carn") })
            x1.update({"carnm": define_G(ge_net_str="carnm") })
            x1.update({"carn_ganm": define_G(ge_net_str="carn_ganm") })
            x1.update({"carn_gan": define_G(ge_net_str="carn_gan") })
            
            opt={"in_channels":3,"out_channels":3,
                "num_features":64,"num_blocks":6,
                "scale":4,
                "num_steps":1,
                "num_groups": 2,   
                "num_layers":3,
                "res_scale":4,
                }
                
            opt.update( {
            "which_model": "D-DBPN",
            "num_features": 64,
            "in_channels": 3,
            "out_channels": 3,
            "num_blocks": 7
            } )
            x1.update({opt["which_model"]:\
             define_G(ge_net_str=opt["which_model"],opt=opt) })
            
            opt.update( {
        "which_model": "EDSR",
        "num_features": 256,
        "in_channels": 3,
        "out_channels": 3,
        "num_blocks": 32,
        "res_scale": 0.1
            } )
            x1.update({
            opt["which_model"]:
             define_G(ge_net_str=opt["which_model"],opt=opt) })
            
            
            
            opt.update( {
        "which_model": "RDN",
        "num_features": 64,
        "in_channels": 3,
        "out_channels": 3,
        "num_blocks": 16,
        "num_layers": 8
            } )
            x1.update({opt["which_model"]:
            define_G(ge_net_str=opt["which_model"],opt=opt) })
            
            
            
            opt.update( {
        "which_model": "SRFBN",
        "num_features": 64,
        "in_channels": 3,
        "out_channels": 3,
        "num_steps": 4,
        "num_groups": 6
            } )
            x1.update({opt["which_model"]: define_G(ge_net_str=opt["which_model"] ,opt=opt) })
            
            return x1 
#         def test_G(self):
#             
#             
#             
#             x1 =self.getnetG_list()
#             with torch.no_grad():
#                 x=torch.randn(1,3,24,24)
#                 for k,netg in x1.items() :
#                     try:
#                         y4=netg(x,scale=4)
#                     except :
#                         y4=netg(x)
#                     if type(y4)==list :
#                         y4 = y4[0]
#                     self.assertEqual(y4.shape ,(1,3,96,96) )
                
            #######test stat_dict 
        def test_pretrained(self):
#             x1 =self.getnetG_list()
            x2 = {
            "SRFBN":"./download_tmp/SRFBN_CVPR19_Models/SRFBN_x4_BI.pth",
            "carnm":"./download_tmp/carn_m.pth",
            "carn":"./download_tmp/carn.pth",
            "carn_gan":"./download_tmp/PCARN-L1.pth",
            "carn_ganm":"./download_tmp/PCARN-M-L1.pth"
            }
            
            x1={}
            x1.update({"carn": define_G(ge_net_str="carn" ,g_path= x2["carn"]) })
            x1.update({"carnm": define_G(ge_net_str="carnm", g_path=x2["carnm"]) })
            x1.update({"carn_ganm": define_G(ge_net_str="carn_ganm", g_path=x2["carn_ganm"]) })
            x1.update({"carn_gan": define_G(ge_net_str="carn_gan",g_path= x2["carn_gan"]) })
            
            opt={"in_channels":3,"out_channels":3,
                "num_features":64,"num_blocks":6,
                "scale":4,
                "num_steps":1,
                "num_groups": 2,   
                "num_layers":3,
                "res_scale":4,
                }
                
            opt.update( {
            "which_model": "D-DBPN",
            "num_features": 64,
            "in_channels": 3,
            "out_channels": 3,
            "num_blocks": 7
            } )
            x1.update({opt["which_model"]:\
             define_G(ge_net_str=opt["which_model"],opt=opt) })
            
            opt.update( {
        "which_model": "EDSR",
        "num_features": 256,
        "in_channels": 3,
        "out_channels": 3,
        "num_blocks": 32,
        "res_scale": 0.1
            } )
            x1.update({
            opt["which_model"]:
             define_G(ge_net_str=opt["which_model"],opt=opt) })
            
            
            
            opt.update( {
        "which_model": "RDN",
        "num_features": 64,
        "in_channels": 3,
        "out_channels": 3,
        "num_blocks": 16,
        "num_layers": 8
            } )
            x1.update({opt["which_model"]:
            define_G(ge_net_str=opt["which_model"],opt=opt) })
            
            
            
            opt.update( {
        "which_model": "SRFBN",
        "num_features": 64,
        "in_channels": 3,
        "out_channels": 3,
        "num_steps": 4,
        "num_groups": 6
            } )
            x1.update({opt["which_model"]: define_G(ge_net_str=opt["which_model"] ,opt=opt,g_path=x2[opt["which_model"]]) })
            
            
            
#             for k ,g_path  in x2.items():
#                 netG= x1[k] 
#                 
#                 if not torch.cuda.is_available():
#                     checkpoint= torch.load(g_path, map_location=lambda storage, loc: storage) 
#                 else :
#                     checkpoint= torch.load(g_path)
#                 if 'state_dict' in checkpoint.keys(): checkpoint = checkpoint['state_dict']
#                 if "module." in list(checkpoint.keys())[0] :
#                     from collections import OrderedDict
#                     new_state_dict = OrderedDict()
#                     for k, v in checkpoint.items():
#                         name = k[7:] # remove `module.`
#                         new_state_dict[name] = v
#                     
#                     checkpoint= new_state_dict
#                 netG.load_state_dict(checkpoint)


#         def test_D(self):
#             d1 = define_D("pixel",input_nc =3 )
#             d2 = define_D("nlayer",input_nc =3 )
#             d3 = define_D("multi",input_nc =3,num_d=4 )
#             
#             print (d1,d2,d3)
#             pass 
        
#         def img_psnr_ssim (self):
#             '''
#             assert with paper 
#             '''
#             pass 
    unittest.main()
