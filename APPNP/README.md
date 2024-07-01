## Run the Code

To run **Propagation then Training Adaptively** (PTA), execute the following command: 

```shell
python train_PT.py --dataset cora_ml --K 10 --alpha 0.1 --dropout 0.0 --epochs 2000 --hidden 64 --lr 0.1 --weight_decay 0.005 --loss_decay 0.05 --fast_mode False --mode 2 --epsilon 100 --str_noise_rate 2.0 --lbl_noise_num 0 --patience 100
```

To run **Propagation then Training Statically** (PTS, a variant of PT), execute the following command: 

```shell
python train_PT.py --dataset cora_ml --K 10 --alpha 0.1 --dropout 0.0 --epochs 2000 --hidden 64 --lr 0.01 --weight_decay 0.005 --loss_decay 0.05 --fast_mode False --mode 0 --epsilon 100 --str_noise_rate 2.0 --lbl_noise_num 0 --patience 100
```

To run **Propagation then Training Dynamically** (PTD, a variant of PT), execute the following command: 

```shell
python train_PT.py --dataset cora_ml --K 10 --alpha 0.1 --dropout 0.0 --epochs 2000 --hidden 64 --lr 0.1 --weight_decay 0.005 --loss_decay 10 --fast_mode False --mode 1 --epsilon 100 --str_noise_rate 2.0 --lbl_noise_num 0 --patience 100
```

To run **APPNP** (our main baseline), execute the following command: 

```shell
python train_APPNP.py --dataset cora_ml --K 10 --alpha 0.1 --dropout 0.5 --epochs 2000 --hidden 64 --lr 0.01 --weight_decay 0.005 --str_noise_rate 2.0 --lbl_noise_num 0 --patience 100
```

- Above commands are for Cora_ML dataset. For Citeseer or Pubmed, replace '--dataset cora_ml' with '--dataset citeseer' or '--dataset pubmed'. For MS_Academic, replace '--dataset cora_ml' with '--dataset ms_academic' and replace '--alpha 0.1' with '--alpha 0.2'. 
- PTS, PTD, and PTA are three variants of PT. The argument '--mode' is for the three variants by setting 0/1/2. PTA performs best among them. 
- The argument '--str_noise_rate' is for specifying structure noise rate in graph. To keep the original graph, set '--str_noise_rate' to 2.0. 
- The argument '--lbl_noise_num' is for specifying label noise numbers of training set. To keep the original labels, set '--lbl_noise_num' to 0. 
