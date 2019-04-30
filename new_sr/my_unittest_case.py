import torch
from models.networks import *
if __name__ == "__main__":
    import  unittest
    import yaml 
    from collections import namedtuple

    class TestAll(unittest.TestCase):

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
            train_config = yaml.load(strx)["train_conf"]
            train_config["dis"] = namedtuple("dis", train_config["dis"].keys())(*train_config["dis"].values() )
            train_config["gen"] = namedtuple("gen", train_config["gen"].keys())(*train_config["gen"].values() )
            train_config = namedtuple("config", train_config.keys())(*train_config.values() )

            self.opt =train_config 
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

#         def test_G(self):
#             g1=define_G(ge_net_str="carn")
#             g2=define_G(ge_net_str="carnm")
#             g3=define_G(ge_net_str="carn_ganm")
#             g4=define_G(ge_net_str="carn_gan")
#             print (g1,g2,g3,g4)
#             with torch.no_grad():
#                 x=torch.randn(1,3,74,74)
#                 y1=g1(x,scale=4)
#                 y2=g2(x,scale=4)
#                 y3=g3(x,scale=4)
#                 y4=g4(x,scale=4)
                
#         def test_D(self):
#             d1 = define_D("pixel",input_nc =3 )
#             d2 = define_D("nlayer",input_nc =3 )
#             d3 = define_D("multi",input_nc =3,num_d=4 )
#             
#             print (d1,d2,d3)
#             pass 
        
        def img_psnr_ssim (self):
            '''
            assert with paper 
            '''
    unittest.main()
