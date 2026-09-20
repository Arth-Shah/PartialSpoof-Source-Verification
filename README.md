# "Where and Who?" — Source Verification for Partial Speech Spoofing

## Overview

This work proposes an **open-set, source-aware spoof localization framework** for partially spoofed speech. Instead of only asking *"is this audio fake?"* or *"where is it fake?"*, we treat each localized spoofed segment as a query and verify its source against a set of reference samples using an embedding-based source verification (SV) approach.

Pipeline (see `docs/`):
1. **Spoof Localization** — a cross-attention BiLSTM predicts frame-level bona fide/spoof labels.
2. **Segment Extraction** — localized spoofed regions are treated as queries.
3. **Source Verification** — a Wavelet Integrated Network (WIN) embedding extractor compares each query segment against enrolled reference embeddings via cosine similarity.

Two conventional acoustic features (MFCC, LFCC) and two self-supervised speech representations (Whisper, HuBERT) are evaluated as frontends.

### Key results

- **HuBERT + BiLSTM** achieves the best localization performance: **17.01% JI_bona** (evaluation set).
- **LFCC + WIN** achieves the best source attribution performance: **21.5% JER_spoof**, a new SOTA for open-set partial spoof source verification.

Full tables and attack-specific breakdowns are in the paper and in `docs/reproducibility.pdf`.

## Repository structure

```
.
├── notebooks/
│   ├── Train/       # Training notebooks, one per frontend feature
│   │   ├── HuBERT-train-final-ps.ipynb
│   │   ├── LFCC-train-final-ps.ipynb
│   │   ├── MFCC-train-final-ps.ipynb
│   │   └── Whisper-train-final-ps.ipynb
│   ├── Dev/         # Development-set evaluation notebooks
│   │   ├── HuBERT-devlopment-final-ps.ipynb
│   │   ├── LFCC-devlopment-final-ps.ipynb
│   │   ├── MFCC-devlopment-final-ps.ipynb
│   │   └── Whisper-devlopment-final-ps.ipynb
│   └── Test/        # Held-out evaluation-set notebooks
│       ├── HuBERT-test-final-ps.ipynb
│       ├── LFCC-test-final-ps.ipynb
│       ├── MFCC-test-final-ps.ipynb
│       └── Whisper-test-final-ps.ipynb
├── docs/
│   └── reproducibility.pdf   # reproducibility material (architecture + dataset details)
├── requirements.txt
├── LICENSE
└── README.md
```

> **Note:** `MFCC-train-final-ps.ipynb` was found alongside the Dev notebooks in the original upload and has been moved into `notebooks/Train/` for consistency. Double check it's the correct file before pushing.

## Dataset

Experiments use a **regenerated version of the PartialSpoof dataset** [Zhang et al., 2022], built on top of ASVspoof 2019 LA. The regeneration follows the original 5-step PartialSpoof construction procedure (normalization, VAD, segment selection, substitution, temporal + balancing), with one addition: the generation-system identity (`gk`) of every substituted spoof segment is retained, to support open-set source attribution. Full construction details are in `docs/reproducibility.pdf`, Section 1.

This dataset is **not included** in this repository — see the [PartialSpoof database](https://doi.org/10.1109/TASLP.2022.3233236) and [ASVspoof 2019](https://www.asvspoof.org/) for access, then follow the construction steps in the reproducibility material to regenerate it with source labels.

## Setup

```bash
git clone https://github.com/Arth-Shah/ICASSP27.git
cd ICASSP27
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Notebooks were developed for GPU-backed environments (Colab / local CUDA). SSL frontends (Whisper, HuBERT) are loaded via 🤗 `transformers` and will download pretrained weights on first run.

## Usage

Each feature/split combination is a self-contained notebook:

1. Run the relevant notebook in `notebooks/Train/` to train the BiLSTM localization model and WIN-based attribution/source-verification model for a given frontend feature.
2. Run the matching notebook in `notebooks/Dev/` or `notebooks/Test/` to evaluate on the development or evaluation split (JI_bona, JER_spoof, attack-specific breakdowns).

Update dataset paths at the top of each notebook to point to your local copy of the regenerated PartialSpoof data.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{shah2027whereandwho,
  title     = {"Where and Who?" Source Verification for Partial Speech Spoofing},
  author    = {Shah, Arth J. and Talari, Sumasri and Namana, Hasini Veni and Patil, Hemant A.},
  booktitle = {Submitted at ICASSP},
  year      = {2027}
}
```

## Generative AI Disclosure

Per the paper, generative AI tools were used for language polishing and bug fixes during preparation of the manuscript and supplementary material; all outputs were manually reviewed and verified by the authors.

## License

See [LICENSE](LICENSE).
