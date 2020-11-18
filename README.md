# bilingual-generative-transformer

Code to train models from "A Bilingual Generative Transformer for Semantic Sentence Embedding". Our code is based on the 58e43cb3ff18f1f47fd62926f00c70cb5920a66f commit Fairseq https://github.com/pytorch/fairseq from Facebook AI Research.

To get started, follow the installation and setup instructions below.

If you use our code for your work please cite:

    @inproceedings{wieting2019beyond,
        title={Beyond BLEU: Training Neural Machine Translation with Semantic Similarity},
        author={Wieting, John and Berg-Kirkpatrick, Taylor and Gimpel, Kevin and Neubig, Graham},
        booktitle={Proceedings of the Association for Computational Linguistics},
        url = {https://arxiv.org/abs/1909.06694},
        year={2019}
    }

Installation and setup instructions:

7. Download and unzip data and semantic similarity models from http://www.cs.cmu.edu/~jwieting.

        wget http://www.cs.cmu.edu/~jwieting/beyond_bleu.zip .
        unzip beyond_bleu.zip
        rm beyond_bleu.zip

To train baseline MLE models in language xx, choices are cs, de, ru, or tr:

    python train.py beyond_bleu/data/data-xx -a fconv_iwslt_de_en --lr 0.25 --clip-norm 0.1 --dropout 0.3 --max-tokens 1000 -s xx -t en --label-smoothing 0.1 --force-anneal 200 --save-dir checkpoints_xx --no-epoch-checkpoints

To train baseline minimum risk models with 1-sBLEU as a cost with alpha=0.3:

    mkdir checkpoints_xx_0.3_word_0.0
    cp beyond_bleu/checkpoints/checkpoints_xx/checkpoint_best.pt checkpoints_xx_0.3_word_0.0/checkpoint_last.pt
    python train.py beyond_bleu/data/data-xx -a fconv_iwslt_de_en --clip-norm 0.1 --momentum 0.9 --lr 0.25 --label-smoothing 0.1 --dropout 0.3 --max-tokens 500 --seq-max-len-a 1.5 --seq-max-len-b 5 --seq-criterion SequenceRiskCriterion --seq-combined-loss-alpha 0.3 --force-anneal 11 --seq-beam 8 --save-dir checkpoints_xx_0.3_word_0.0 --seq-score-alpha 0 -s xx -t en --reset-epochs

To train baseline minimum risk models with 1-SimiLe as a cost with alpha=0.3:

    mkdir checkpoints_xx_0.3_word_1.0
    cp beyond_bleu/checkpoints/checkpoints_xx/checkpoint_best.pt checkpoints_xx_0.3_word_1.0/checkpoint_last.pt
    python train.py beyond_bleu/data/data-xx -a fconv_iwslt_de_en --clip-norm 0.1 --momentum 0.9 --lr 0.25 --label-smoothing 0.1 --dropout 0.3 --max-tokens 500 --seq-max-len-a 1.5 --seq-max-len-b 5 --seq-criterion SequenceRiskCriterion --seq-combined-loss-alpha 0.3 --force-anneal 11 --seq-beam 8 --save-dir checkpoints_xx_0.3_word_1.0 --seq-score-alpha 1 -s xx -t en --sim-model-file beyond_bleu/sim/sim.pt --reset-epochs

To evaluate models in terms of corpus BLEU, SIM, and SimiLe:

    python evaluate.py --data beyond_bleu/data/data-xx -s xx -t en --save-dir checkpoints_xx_0.3_word_1.0 --length_penalty 0.25 --sim-model-file beyond_bleu/sim/sim.pt