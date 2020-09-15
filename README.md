<br/>
<h1 align="center">Cross-Domain Relation Adaptation</h1>
<br/>

<br/>

This is the code repository for the paper "Cross-Domain Relation Adaptation". The repository contains the code for the three experiments presented in the paper: transformers on PPI and GLACE and GCN on citation graphs.

## Abstract

We consider the problem of learning the relation between a sample in domain A and a sample in domain B, based on supervised samples of the relation between pairs of samples in A and pairs of samples in B. In other words, we present a semi-supervised setting in which there are no labeled mixed-domain pairs of samples. 
Our method is derived based on a generalization bound and includes supervised terms in each domain, a domain confusion term on the learned features, and a consistency term between the domain-specific relations when considering mixed pairs. 
Our results demonstrate the effectiveness of our method in two very different domains: (i) prediction of protein-protein interactions between viruses and hosts based on modeling sequences, and (ii) link prediction in citation graphs using graph neural networks.

## Usage

### GCN

Our method implemented for citation networks using [GCN](https://arxiv.org/abs/1609.02907). The implementation uses [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) as its base, splitting the codebase into the training and implementation routine seamlessly. We use Pytorch lightning Trainer's routine for the task, which provides all the extra options that the library provides (for more information run the command with `--help`). The specific parameters related to our method is described in the following command:

```
python train.py \
    --dataset citation_citeseer|plaintoid_cora|citation_dblp|citation_pubmed \ # The citation network to use.
    --epoch_size 50 \ # Number of batch per epoch.
    --lr 2e-4 \ # The learning rate to use.
    --supervised True \ # Whether to use the cross samples supervisedly.
    --multiheads True \ # Whether to use multiple classifiers for each group.
    --hidden_c 128 \ # The representation size after the first layer (backbone).
    --out_c 64 \ # The embeding dim size.
    --co_coef 1.0 \ # The domain confusion term coefficient. (lambda_1)
    --da_coef 1.0  # The domain confusion term coefficient. (lambda_2)
```

Please refer to [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) for all additional options.

### GLACE

Our method implemented for citation networks on top of the original implementation of [GLACE](https://arxiv.org/pdf/1912.00536v1.pdf) using the TensorBoard V1 [GLACE](https://github.com/bhagya-hettige/GLACE). Run the training routine in order to test the method using the following command:

```
python train.py cora|cora_ml|dblp|citeseer \ # Dataset name (npz file located at reletive data folder with the same name)
                --embedding_dim 64 \ # The embedding dim size.
                --batch_size 128 \ # The batch size.
                --learning_rate 1e-3 \ # The learning rate.
                --learning_rate_bb 1e-3 \ # learning rate of the first layer only (backbone).
                --co_coef 1.0 \ # The consistency term coefficient. (lambda_1)
                --da_coef 1.0 \ # The domain confusion term coefficient. (lambda_2)
                --use_multihead \ # Whether to use classifier per class.
                --supervised # Whether to learn on the crosses supervisedly.
```

### PPI

Our method implemented for inter-species protein-protein interactions in virus-hosts using intra-species interactions in viruses and hosts. The four backbone available are [TAPE](https://arxiv.org/pdf/1906.08230.pdf) from its [public repository](https://github.com/songlab-cal/tape), and ProtBert\ProtXLNet\ProtAlbert implemented using [huggingface Transformers](https://github.com/huggingface/transformers) and trained by [ProtTrans](https://www.biorxiv.org/content/10.1101/2020.07.12.199554v2) (The code automatically download the weights from the [public repository](https://github.com/agemagician/ProtTrans)). The dataset used is extracted from [Viruses.STRING](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6213343/) (available to download using the [site](http://viruses.string-db.org/cgi/download.pl?UserId=V8TlnL2PVMTy&sessionId=lZxVON19FJop)), and is filtered for proteins who share more than 25% sequence identity. The implementation uses [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) as its base, splitting the codebase into the training and implementation routine seamlessly. We use Pytorch lightning Trainer's routine for the task, which provides all the extra options that the library provides (for more information run the command with `--help`). The specific parameters related to our method is described in the following command:

```
python  train.py 
        --bb tape|ptra_bert|ptra_albert|ptra_xlnet \ # The network to use as the backbone for the method. 
                                                   \ #Tape being tape and ptra_XX being ProtTrans XXX.
        --lr 2e-4 \ # The learning rate
        --bb_lr 1e-6  \ # Te learning rate for the backbone.
        --co_coef 1.0 \ # The consistency term coefficient. (lambda_1)
        --da_coef 1.0 \ # The domain confusion term coefficient. (lambda_2)
        --max_length 500 \ # The maximum length of protein to use. Proteins who exceed it will be truncated.
        --train_from_scratch \ # Whther to run load the pretrained weights of the backbone,
                             \ # or to re-initialize it.
        --dont_use_multipleheads \ # Whether to use two heads for viruses and host, 
                                 \ # or a single head for both.
        --batch_size 3 \ # Batch size to use.
        --epoch_size 8000 \ # Epoch size to use.
        --batch_size_val 45 \ # Batch size for val\test (as it require less memory, can be larger.
        --epoch_size_val 9000 \ # Epoch size for val\test. 
        --data_root ./STRING # Root folder of the STRING database.
```

Please refer to [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) for all additional options.
