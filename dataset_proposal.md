# Project 3, Path B: Dataset proposal

Group: AS (Arian and Simon)

Dataset: Gravity Spy training set v1.0 (Bahaadini et al. 2018). It contains time–frequency spectrograms of transient noise "glitches" from the Advanced LIGO detectors in the O1 and O2 observing runs, labelled by Gravity Spy volunteers on Zooniverse.

## Download, format and licence

- Zenodo, no registration needed: <https://zenodo.org/records/1486046>.
- Format: PNGs or HDF5 file with images, labels and split assignment, plus a CSV metadata file, we will use HDF5.
- Licence: Creative Commons Attribution 4.0 International

## Size

- 8,583 labelled glitches in 22 morphological classes, an issue could be that some of the classes has few samples, but thats more fun
- Each sample has four views (0.5, 1, 2 and 4 s windows). Each view is a 140x170 grey-scale image, i.e. 23,800 pixels per view.
- We do understand that the size, both in number of samples and in number of pixels, could make it more difficult, but we will use our access to GPUs and nodes available to use from the Institute for Theoretical AStrophysics for all our means.

## Train/test split

- The dataset comes with a pre-split: 70% train, 15% validation, 15% test.
- We will ofc tune hyperparameters on the training and validation sets only, and use the test set once, at the end.

## Published analyses to compare with

- Zevin et al., Class. Quantum Grav. 34, 064003 (2017), [arXiv:1611.04596](https://arxiv.org/abs/1611.04596): the Gravity Spy project.
- Bahaadini et al., Information Sciences 444, 172 (2018): the dataset paper, with multi-view CNN benchmarks.
- George, Shen & Huerta (2017), [arXiv:1706.07446](https://arxiv.org/abs/1706.07446): deep transfer learning on the same data.
- Later CNN and attention-based studies, e.g. [arXiv:2303.13917](https://arxiv.org/abs/2303.13917) and [arXiv:2401.12913](https://arxiv.org/abs/2401.12913).
- these citations are found and partially written by OpenAI chatgpt, then manually verified by us

## Why the course methods fit

- Image data, CNN will be used as the main model, additioanlly we will compare one view against all four views stacked as channels.
- Further (as we understand), Baselines:
  - logistic regression on raw pixels or PCA features;
  - a feed-forward network on the flattened, downsampled images;
  - gradient boosting / random forest on downsampled or PCA features.

- We do understand we still lack some knowledge of the data set and how this problem will differ, but we are very exited for this as its relevant for other courses aswell