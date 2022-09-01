#GCN
python train.py --dataset Cora --encoder GCN --encoder_type 3 --sample_size 0.6 --debias 0.12
python train.py --dataset CiteSeer --encoder GCN --encoder_type 3 --sample_size 1 --debias 0.11
python train.py --dataset PubMed --encoder GCN --encoder_type 2 --sample_size 0.2 --debias 0.02
python train_coauthor.py --dataset Coauthor-CS --encoder GCN --encoder_type 2 --sample_size 0.3 --debias 0.07 --epoch 700
python train_coauthor.py --dataset Coauthor-Phy --encoder GCN --encoder_type 2 --sample_size 0.01 --debias 0.1

#SGC
python train.py --dataset Cora --encoder SGC --sample_size 0.05 --debias 0.13
python train.py --dataset CiteSeer --encoder SGC --sample_size 0.2 --debias 0.
python train.py --dataset PubMed --encoder SGC --sample_size 0.5 --debias 0.02
python train_coauthor.py --dataset Coauthor-CS --encoder SGC --sample_size 0.05 --debias 0.0 --epoch 700
python train_coauthor.py --dataset Coauthor-Phy --encoder SGC --sample_size 0.01 --debias 0.05

#APPNP
python train_APPNP_cl.py --dataset Cora --K 10 --alpha 0.1 --dropout 0.5 --epochs 2000 --hidden 64 --lr 0.01 --weight_decay 0.005 --str_noise_rate 2.0 --lbl_noise_num 0 --patience 100 --sample_size 0.05 --debias 0.04
python train_APPNP_cl.py --dataset CiteSeer --K 10 --alpha 0.1 --dropout 0.5 --epochs 2000 --hidden 64 --lr 0.01 --weight_decay 0.005 --str_noise_rate 2.0 --lbl_noise_num 0 --patience 100 --sample_size 0.2 --debias 0.1
python train_APPNP_cl.py --dataset PubMed --K 10 --alpha 0.1 --dropout 0.5 --epochs 2000 --hidden 64 --lr 0.01 --weight_decay 0.005 --str_noise_rate 2.0 --lbl_noise_num 0 --patience 100 --sample_size 1 --debias 0.01
python train_APPNP_coauthor.py --dataset Coauthor-CS --K 4 --alpha 0.1 --dropout 0.5 --epochs 2000 --hidden 64 --lr 0.05 --weight_decay 0.005 --str_noise_rate 2.0 --lbl_noise_num 0 --patience 100 --sample_size 0.4 --debias 0.
python train_APPNP_coauthor.py --dataset Coauthor-Phy --K 4 --alpha 0.1 --dropout 0.5 --epochs 2000 --hidden 64 --lr 0.05 --weight_decay 0.005 --str_noise_rate 2.0 --lbl_noise_num 0 --patience 100 --sample_size 0.01 --debias 0.12

#GAT
python train_dgl_cl.py --dataset cora --hidden 16 --sample_size 0.01 --debias 0.08 --gpu 2
python train_dgl_cl.py --dataset citeseer --hidden 16 --sample_size 0.5 --debias 0.13 --gpu 2
python train_dgl_cl.py --dataset pubmed --hidden 16 --out_heads 8 --weight_decay 0.001 --sample_size 0.5 --debias 0.05 --dropout1 0. --gpu 2
python train_dgl_cl_coauthor.py --dataset coauthor-cs --hidden 16 --out_heads 8 --weight_decay 0.005 --sample_size 0.3 --debias 0.11 --gpu 2
python train_dgl_cl_coauthor.py --dataset coauthor-phy --hidden 16 --out_heads 8 --weight_decay 0.005 --sample_size 0.1 --debias 0.07 --gpu 2

##消融实验
##全负例
python train.py --dataset Cora --encoder GCN --encoder_type 3 --sample_size 0.3 --neg_type 0.3 --weight 0.8 --debias 0.
python train.py --dataset CiteSeer --encoder GCN --encoder_type 3 --sample_size 0.8 --neg_type 0.3 --weight 0.2 --debias 0.

####速度
python train.py --dataset Cora --encoder GCN --encoder_type 3 --sample_size 0.1 --debias 0.13
python train.py --dataset PubMed --encoder GCN --encoder_type 2 --sample_size 0.05 --debias 0.0


