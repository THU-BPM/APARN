# APARN
Code and datasets of our paper "[AMR-based Network for Aspect-based Sentiment Analysis](https://aclanthology.org/2023.acl-long.19/)" accepted by ACL 2023.

## Requirements

- torch>=1.13.1
- scikit-learn==0.23.2
- transformers==3.2.0
- nltk==3.5
- einops==0.4.1

To install requirements, run `pip install -r requirements.txt`.

## Training

To train and evaluate the APARN model, run:

`./APARN/run.sh`

If having trouble downloading *.pt with git-lfs, you can use the following link instead: [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/79874f22fcb04b6b9842/?dl=1)

## Credits

The code and datasets in this repository are based on [DualGCN](https://github.com/CCChenhao997/DualGCN-ABSA) and [Alphafold2-Pytorch](https://github.com/lucidrains/alphafold2).

## Citation

If you find this work useful, please cite as following.

```
@inproceedings{ma-etal-2023-amr,
    title = "{AMR}-based Network for Aspect-based Sentiment Analysis",
    author = "Ma, Fukun  and
      Hu, Xuming  and
      Liu, Aiwei  and
      Yang, Yawen  and
      Li, Shuang  and
      Yu, Philip S.  and
      Wen, Lijie",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.19",
    doi = "10.18653/v1/2023.acl-long.19",
    pages = "322--337",
    abstract = "Aspect-based sentiment analysis (ABSA) is a fine-grained sentiment classification task. Many recent works have used dependency trees to extract the relation between aspects and contexts and have achieved significant improvements. However, further improvement is limited due to the potential mismatch between the dependency tree as a syntactic structure and the sentiment classification as a semantic task. To alleviate this gap, we replace the syntactic dependency tree with the semantic structure named Abstract Meaning Representation (AMR) and propose a model called AMR-based Path Aggregation Relational Network (APARN) to take full advantage of semantic structures. In particular, we design the path aggregator and the relation-enhanced self-attention mechanism that complement each other. The path aggregator extracts semantic features from AMRs under the guidance of sentence information, while the relation-enhanced self-attention mechanism in turn improves sentence features with refined semantic information. Experimental results on four public datasets demonstrate 1.13{\%} average F1 improvement of APARN in ABSA when compared with state-of-the-art baselines.",
}
```
