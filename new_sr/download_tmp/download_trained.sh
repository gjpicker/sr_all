
echo ""
echo ""
echo "******************  Network ******************"
bash ../utils/gdrive_download 1Dsb_-OH0CeSJVjvP9A4bh2_IBQh9R-ja ./trained.tar 
tar -xvf ./trained.tar 



echo ""
echo ""
echo "******************  Network ******************"
wget https://github.com/nmhkahn/CARN-pytorch/raw/master/checkpoint/carn_m.pth?raw=True -O carn_m.pth 
wget https://github.com/nmhkahn/CARN-pytorch/raw/master/checkpoint/carn.pth?raw=True -O carn.pth 



echo ""
echo ""
echo "******************  Network ******************"
wget https://github.com/nmhkahn/PCARN-pytorch/blob/master/checkpoints/PCARN-L1.pth?raw=true -O PCARN-L1.pth 
wget https://github.com/nmhkahn/PCARN-pytorch/blob/master/checkpoints/PCARN-M-L1.pth?raw=true -O PCARN-M-L1.pth 



