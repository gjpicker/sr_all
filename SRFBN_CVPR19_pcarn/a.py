import networks 


opt = {"scale":4}
opt .update({
        "which_model": "RDN",
                "num_features": 64,
                        "in_channels": 3,
                                "out_channels": 3,
                                        "num_blocks": 16,
                                                "num_layers": 8
                                                    })



import torch
net = networks.define_net(opt)
print (net)


x=torch.randn(1,3,40,40)
ret =  net(x)

print ("ype:",type(ret),len(ret))
for t in ret :
    print (t.shape)

