# DADS model evaluation

Generated: 2026-10-03T06:30:10+00:00

Dataset: {'labels_csv': 'dataset/SEP-28k_labels.csv', 'waves_dir': 'dataset/Waves', 'clips_used': 26716, 'clips_total': 28177, 'dropped': {'Music': 396, 'NoSpeech': 856, 'empty_audio': {'HeStutters': 110, 'StrongVoices': 34, 'StutterTalk': 7, 'WomenWhoStutter': 58}}}

Decision threshold: 0.4

## Split: random

```
{'name': 'random', 'group_by': None, 'sizes': {'train': 18701, 'val': 4007, 'test': 4008}, 'seed': 42, 'test_size': 0.15, 'val_size': 0.15, 'strategy': 'multi-label stratified random (iterative_train_test_split)', 'leakage': 'speaker identity leaks between splits — scores are optimistic', 'label_distribution': {'train': {'prolongation': 0.3077, 'block': 0.4265, 'soundrep': 0.1978, 'wordrep': 0.1691, 'interjection': 0.3502}, 'val': {'prolongation': 0.3077, 'block': 0.4268, 'soundrep': 0.1979, 'wordrep': 0.169, 'interjection': 0.3501}, 'test': {'prolongation': 0.3076, 'block': 0.4264, 'soundrep': 0.1979, 'wordrep': 0.1692, 'interjection': 0.3503}}}
```

Threshold: 0.40

| Model | AUROC | AUPRC | Precision | Recall | F1 | Specificity | Balanced Acc | Accuracy | Positives | n |
|---|---|---|---|---|---|---|---|---|---|---|
| prolongation | 0.7495 | 0.6056 | 0.4927 | 0.6537 | 0.5619 | 0.7009 | 0.6773 | 0.6864 | 1233 | 4008 |
| block | 0.6535 | 0.5714 | 0.4374 | 0.9778 | 0.6044 | 0.0652 | 0.5215 | 0.4543 | 1709 | 4008 |
| soundrep | 0.6429 | 0.3363 | 0.2711 | 0.6318 | 0.3794 | 0.5810 | 0.6064 | 0.5911 | 793 | 4008 |
| wordrep | 0.6635 | 0.2753 | 0.3917 | 0.1254 | 0.1899 | 0.9604 | 0.5429 | 0.8191 | 678 | 4008 |
| interjection | 0.6364 | 0.4915 | 0.4133 | 0.7251 | 0.5265 | 0.4451 | 0.5851 | 0.5432 | 1404 | 4008 |
| **macro** | 0.6692 | — | — | — | **0.4524** | — | — | — | — | — |

| Model | F1 @ 0.4 | Best F1 | Best threshold | ΔF1 |
|---|---|---|---|---|
| prolongation | 0.5619 | 0.5635 | 0.35 | +0.0016 |
| block | 0.6044 | 0.6099 | 0.50 | +0.0055 |
| soundrep | 0.3794 | 0.3794 | 0.40 | +0.0000 |
| wordrep | 0.1899 | 0.3458 | 0.25 | +0.1559 |
| interjection | 0.5265 | 0.5337 | 0.35 | +0.0072 |


Joint view — exact-match accuracy: **0.0828**, Hamming loss: **0.3812**

## Split: group_episode

```
{'name': 'group_episode', 'group_by': 'episode', 'sizes': {'train': 18453, 'val': 4070, 'test': 4193}, 'seed': 42, 'test_size': 0.15, 'val_size': 0.15, 'num_groups': 384, 'strategy': 'group-disjoint by episode', 'test_groups': ['HeStutters|12', 'HeStutters|18', 'HeStutters|20', 'HeStutters|21', 'HeStutters|6', 'StrongVoices|10', 'StrongVoices|23', 'StutterTalk|43', 'StutterTalk|6', 'WomenWhoStutter|101', 'WomenWhoStutter|21', 'WomenWhoStutter|39', 'WomenWhoStutter|42', 'WomenWhoStutter|50', 'WomenWhoStutter|62', 'WomenWhoStutter|88', 'WomenWhoStutter|99'], 'val_groups': ['HeStutters|11', 'HeStutters|2', 'HeStutters|22', 'HeStutters|23', 'MyStutteringLife|35', 'StrongVoices|6', 'StutterTalk|33', 'StutterTalk|68', 'StutteringIsCool|80', 'WomenWhoStutter|10', 'WomenWhoStutter|14', 'WomenWhoStutter|19', 'WomenWhoStutter|2', 'WomenWhoStutter|30', 'WomenWhoStutter|34', 'WomenWhoStutter|57', 'WomenWhoStutter|6', 'WomenWhoStutter|74', 'WomenWhoStutter|87', 'WomenWhoStutter|97'], 'label_distribution': {'train': {'prolongation': 0.2858, 'block': 0.4247, 'soundrep': 0.1901, 'wordrep': 0.1606, 'interjection': 0.3458}, 'val': {'prolongation': 0.3447, 'block': 0.459, 'soundrep': 0.2376, 'wordrep': 0.2002, 'interjection': 0.3845}, 'test': {'prolongation': 0.3682, 'block': 0.4031, 'soundrep': 0.1934, 'wordrep': 0.1762, 'interjection': 0.3365}}}
```

Threshold: 0.40

| Model | AUROC | AUPRC | Precision | Recall | F1 | Specificity | Balanced Acc | Accuracy | Positives | n |
|---|---|---|---|---|---|---|---|---|---|---|
| prolongation | 0.7298 | 0.6203 | 0.5245 | 0.7014 | 0.6002 | 0.6293 | 0.6654 | 0.6559 | 1544 | 4193 |
| block | 0.5986 | 0.4902 | 0.4157 | 0.9343 | 0.5754 | 0.1135 | 0.5239 | 0.4443 | 1690 | 4193 |
| soundrep | 0.5693 | 0.2444 | 0.2291 | 0.5364 | 0.3210 | 0.5671 | 0.5517 | 0.5612 | 811 | 4193 |
| wordrep | 0.6392 | 0.2660 | 0.3364 | 0.1461 | 0.2038 | 0.9383 | 0.5422 | 0.7987 | 739 | 4193 |
| interjection | 0.6229 | 0.4654 | 0.3770 | 0.7647 | 0.5050 | 0.3591 | 0.5619 | 0.4956 | 1411 | 4193 |
| **macro** | 0.6320 | — | — | — | **0.4411** | — | — | — | — | — |

| Model | F1 @ 0.4 | Best F1 | Best threshold | ΔF1 |
|---|---|---|---|---|
| prolongation | 0.6002 | 0.6050 | 0.30 | +0.0048 |
| block | 0.5754 | 0.5773 | 0.45 | +0.0019 |
| soundrep | 0.3210 | 0.3296 | 0.25 | +0.0086 |
| wordrep | 0.2038 | 0.3517 | 0.30 | +0.1479 |
| interjection | 0.5050 | 0.5165 | 0.35 | +0.0115 |


Joint view — exact-match accuracy: **0.0625**, Hamming loss: **0.4089**

## Split comparison

delta = random minus group_episode; a large positive delta indicates speaker leakage

| Model | F1 (random) | F1 (group_episode) | dF1 | AUROC (random) | AUROC (group_episode) | dAUROC |
|---|---|---|---|---|---|---|
| prolongation | 0.5619 | 0.6002 | -0.0383 | 0.7495 | 0.7298 | +0.0197 |
| block | 0.6044 | 0.5754 | +0.0290 | 0.6535 | 0.5986 | +0.0549 |
| soundrep | 0.3794 | 0.3210 | +0.0584 | 0.6429 | 0.5693 | +0.0736 |
| wordrep | 0.1899 | 0.2038 | -0.0139 | 0.6635 | 0.6392 | +0.0243 |
| interjection | 0.5265 | 0.5050 | +0.0215 | 0.6364 | 0.6229 | +0.0135 |
