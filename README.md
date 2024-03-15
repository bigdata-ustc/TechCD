# TechCD: Leveraging Transferable Knowledge Concept Graph Embedding for Cold-Start Cognitive Diagnosis

This repository contains the implementation for the paper titled **Leveraging Transferable Knowledge Concept Graph Embedding for Cold-Start Cognitive Diagnosis**, which was published at `SIGIR'2023`. [[Paper](https://dl.acm.org/doi/10.1145/3539618.3591774)][[Presentation Video](https://dl.acm.org/action/downloadSupplement?doi=10.1145%2F3539618.3591774&file=SIGIR23-fp1870.mp4)]

Authors: [Weibo Gao](https://scholar.google.com/citations?user=k19RS74AAAAJ&hl=zh-CN), [Hao Wang](http://staff.ustc.edu.cn/~wanghao3), [Qi Liu](http://staff.ustc.edu.cn/~qiliuql) et al.

Email: weibogao@mail.ustc.edu.cn


> Tips: The code and documentation will be completed by April.

## Environment Settings
Install the Python dependencies:
```
pip install -r requirements.txt
```

Install the CUDA version of DGL:
- Download the DGL-Cuda package from [here](https://data.dgl.ai/wheels/cu118/repo.html).
- Install the DGL-Cuda package (e.g., `dgl-1.1.2+cu118-cp311-cp311-win_amd64.whl`) using `pip install dgl-1.1.2+cu118-cp311-cp311-win_amd64.whl`.

## Codes & Datasets (Junyi & ASSIST)
To train TechCD:
```
python train.py
```

## Related Works
- **RCD: Relation Map Driven Cognitive Diagnosis for Intelligent Education Systems (SIGIR'2021).** [[Paper](https://dl.acm.org/doi/abs/10.1145/3404835.3462932)][[Code](https://github.com/bigdata-ustc/RCD/)][[Presentation Video](https://dl.acm.org/action/downloadSupplement?doi=10.1145%2F3404835.3462932&file=RCD.mp4)]
- **Zero-1-to-3: Domain-level Zero-shot Cognitive Diagnosis via One Batch of Early-bird Students towards Three Diagnostic Objectives (AAAI'2024).** [[Paper](https://arxiv.org/abs/2312.13434)][[Code](https://github.com/bigdata-ustc/Zero-1-to-3)]


## BibTex
Please cite this paper if you use our codes. Thanks!
```
@inproceedings{gao2023leveraging,
  title={Leveraging transferable knowledge concept graph embedding for cold-start cognitive diagnosis},
  author={Gao, Weibo and Wang, Hao and Liu, Qi and Wang, Fei and Lin, Xin and Yue, Linan and Zhang, Zheng and Lv, Rui and Wang, Shijin},
  booktitle={Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={983--992},
  year={2023}
}
```

## Update Logs

---
- March 16, 2024: Updated codes
- March 14, 2024: Added codes and data examples
- August 29, 2023: Created this project

