# DADS model evaluation

Generated: 2026-10-03T06:29:35+00:00

Dataset: {'labels_csv': 'dataset/SEP-28k_labels.csv', 'waves_dir': 'dataset/Waves', 'clips_used': 26716, 'clips_total': 28177, 'dropped': {'Music': 396, 'NoSpeech': 856, 'empty_audio': {'HeStutters': 110, 'StrongVoices': 34, 'StutterTalk': 7, 'WomenWhoStutter': 58}}}

Decision threshold: 0.4

## Split: random

```
{'name': 'random', 'group_by': None, 'sizes': {'train': 18701, 'val': 4007, 'test': 4008}, 'seed': 42, 'test_size': 0.15, 'val_size': 0.15, 'strategy': 'multi-label stratified random (iterative_train_test_split)', 'leakage': 'speaker identity leaks between splits — scores are optimistic', 'label_distribution': {'train': {'prolongation': 0.3077, 'block': 0.4265, 'soundrep': 0.1978, 'wordrep': 0.1691, 'interjection': 0.3502}, 'val': {'prolongation': 0.3077, 'block': 0.4268, 'soundrep': 0.1979, 'wordrep': 0.169, 'interjection': 0.3504}, 'test': {'prolongation': 0.3076, 'block': 0.4264, 'soundrep': 0.1979, 'wordrep': 0.1692, 'interjection': 0.35}}}
```

> **Production weights were trained on an unseeded random ~70% of every clip (Model/model_train.ipynb), so their test subset overlaps training data. Treat these numbers as optimistic and retrain with `python -m ml.train` for clean scores.**

Threshold: 0.40

| Model | AUROC | AUPRC | Precision | Recall | F1 | Specificity | Balanced Acc | Accuracy | Positives | n |
|---|---|---|---|---|---|---|---|---|---|---|
| prolongation | 0.7304 | 0.5765 | 0.7610 | 0.1549 | 0.2574 | 0.9784 | 0.5666 | 0.7250 | 1233 | 4008 |
| block | 0.6656 | 0.5791 | 0.5595 | 0.6132 | 0.5851 | 0.6411 | 0.6272 | 0.6292 | 1709 | 4008 |
| soundrep | 0.6749 | 0.3603 | 0.4032 | 0.2207 | 0.2852 | 0.9194 | 0.5701 | 0.7812 | 793 | 4008 |
| wordrep | 0.7440 | 0.3953 | 0.5543 | 0.1431 | 0.2274 | 0.9766 | 0.5598 | 0.8356 | 678 | 4008 |
| interjection | 0.6811 | 0.5536 | 0.5893 | 0.3550 | 0.4431 | 0.8668 | 0.6109 | 0.6876 | 1403 | 4008 |
| **macro** | 0.6992 | — | — | — | **0.3596** | — | — | — | — | — |

| Model | F1 @ 0.4 | Best F1 | Best threshold | ΔF1 |
|---|---|---|---|---|
| prolongation | 0.2574 | 0.5654 | 0.10 | +0.3080 |
| block | 0.5851 | 0.6188 | 0.30 | +0.0337 |
| soundrep | 0.2852 | 0.3958 | 0.20 | +0.1106 |
| wordrep | 0.2274 | 0.4269 | 0.10 | +0.1995 |
| interjection | 0.4431 | 0.5567 | 0.15 | +0.1136 |


Joint view — exact-match accuracy: **0.2188**, Hamming loss: **0.2683**

## Split: group_episode

```
{'name': 'group_episode', 'group_by': 'episode', 'sizes': {'train': 18453, 'val': 4070, 'test': 4193}, 'seed': 42, 'test_size': 0.15, 'val_size': 0.15, 'num_groups': 384, 'strategy': 'group-disjoint by episode', 'test_groups': ['HeStutters|12', 'HeStutters|18', 'HeStutters|20', 'HeStutters|21', 'HeStutters|6', 'StrongVoices|10', 'StrongVoices|23', 'StutterTalk|43', 'StutterTalk|6', 'WomenWhoStutter|101', 'WomenWhoStutter|21', 'WomenWhoStutter|39', 'WomenWhoStutter|42', 'WomenWhoStutter|50', 'WomenWhoStutter|62', 'WomenWhoStutter|88', 'WomenWhoStutter|99'], 'val_groups': ['HeStutters|11', 'HeStutters|2', 'HeStutters|22', 'HeStutters|23', 'MyStutteringLife|35', 'StrongVoices|6', 'StutterTalk|33', 'StutterTalk|68', 'StutteringIsCool|80', 'WomenWhoStutter|10', 'WomenWhoStutter|14', 'WomenWhoStutter|19', 'WomenWhoStutter|2', 'WomenWhoStutter|30', 'WomenWhoStutter|34', 'WomenWhoStutter|57', 'WomenWhoStutter|6', 'WomenWhoStutter|74', 'WomenWhoStutter|87', 'WomenWhoStutter|97'], 'label_distribution': {'train': {'prolongation': 0.2858, 'block': 0.4247, 'soundrep': 0.1901, 'wordrep': 0.1606, 'interjection': 0.3458}, 'val': {'prolongation': 0.3447, 'block': 0.459, 'soundrep': 0.2376, 'wordrep': 0.2002, 'interjection': 0.3845}, 'test': {'prolongation': 0.3682, 'block': 0.4031, 'soundrep': 0.1934, 'wordrep': 0.1762, 'interjection': 0.3365}}}
```

> **Production weights were trained on an unseeded random ~70% of every clip (Model/model_train.ipynb), so their test subset overlaps training data. Treat these numbers as optimistic and retrain with `python -m ml.train` for clean scores.**

Threshold: 0.40

| Model | AUROC | AUPRC | Precision | Recall | F1 | Specificity | Balanced Acc | Accuracy | Positives | n |
|---|---|---|---|---|---|---|---|---|---|---|
| prolongation | 0.7199 | 0.6016 | 0.6662 | 0.2856 | 0.3998 | 0.9166 | 0.6011 | 0.6842 | 1544 | 4193 |
| block | 0.6350 | 0.5347 | 0.5174 | 0.5018 | 0.5095 | 0.6840 | 0.5929 | 0.6105 | 1690 | 4193 |
| soundrep | 0.6552 | 0.3092 | 0.3904 | 0.0900 | 0.1463 | 0.9663 | 0.5282 | 0.7968 | 811 | 4193 |
| wordrep | 0.8069 | 0.4986 | 0.6188 | 0.1867 | 0.2869 | 0.9754 | 0.5811 | 0.8364 | 739 | 4193 |
| interjection | 0.6748 | 0.5278 | 0.5068 | 0.5025 | 0.5046 | 0.7520 | 0.6272 | 0.6680 | 1411 | 4193 |
| **macro** | 0.6984 | — | — | — | **0.3694** | — | — | — | — | — |

| Model | F1 @ 0.4 | Best F1 | Best threshold | ΔF1 |
|---|---|---|---|---|
| prolongation | 0.3998 | 0.5985 | 0.10 | +0.1987 |
| block | 0.5095 | 0.5862 | 0.20 | +0.0767 |
| soundrep | 0.1463 | 0.3777 | 0.20 | +0.2314 |
| wordrep | 0.2869 | 0.5201 | 0.15 | +0.2332 |
| interjection | 0.5046 | 0.5327 | 0.15 | +0.0281 |


Joint view — exact-match accuracy: **0.1882**, Hamming loss: **0.2808**

## Split comparison

delta = random minus group_episode; a large positive delta indicates speaker leakage

| Model | F1 (random) | F1 (group_episode) | dF1 | AUROC (random) | AUROC (group_episode) | dAUROC |
|---|---|---|---|---|---|---|
| prolongation | 0.2574 | 0.3998 | -0.1424 | 0.7304 | 0.7199 | +0.0105 |
| block | 0.5851 | 0.5095 | +0.0756 | 0.6656 | 0.6350 | +0.0306 |
| soundrep | 0.2852 | 0.1463 | +0.1389 | 0.6749 | 0.6552 | +0.0197 |
| wordrep | 0.2274 | 0.2869 | -0.0595 | 0.7440 | 0.8069 | -0.0629 |
| interjection | 0.4431 | 0.5046 | -0.0615 | 0.6811 | 0.6748 | +0.0063 |
