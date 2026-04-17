## CoPRIME

Implementation of the CoPRIME (Contrastive Probabilistic Routing for IMbalanced tokens with ELBO-regularized mixture of experts) paper.

## Usage

Pre-train:
```
python main.py pretrain
```

Fine-tuning:
```
python main.py finetune --checkpoint ./checkpoints/ptrtrain_epoch1.pt
```

Evaluation:
```
python main.py evaluate --checkpoint ./checkpoints/finetune_epoch1.pt --dataset iemocap
python main.py evaluate --checkpoint ./checkpoints/finetune_epoch1.pt --dataset mosei
```


## Citation
This work has been accepted in AISTATS 2026, Citation:
```
@InProceedings{pmlr-naderi2026coprime,
  title =       {From Token Imbalance to Balanced Routing: An ELBO-Regularized Probabilistic Framework for Contrastive Multimodal Learning},
  author =      {Naderi, Habibeh and Haji Soleimani, Behrouz and Matwin, Stan},
  booktitle =   {29th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year =        {2026}
}
```

## License
[MIT License](LICENSE)
