# VET_privacy
## Setup
### Installation des dépendances pour le projet
```bash
./installRequirements.sh
```

## Dépendances
- **Python :** 3.4, 3.5 ou 3.6
- **Dépendances python :**
    - scipy >= 0.17
    - mpmath
    - pandas
    - numpy


### Installation Tensorflow
```bash
# Current release for CPU-only
$ pip install tensorflow

# Nightly build for CPU-only (unstable)
$ pip install tf-nightly

# GPU package for CUDA-enabled GPU cards
$ pip install tensorflow-gpu

# Nightly build with GPU support (unstable)
$ pip install tf-nightly-gpu
```

### Installation de la librairie privacy
```bash
$ git clone https://github.com/tensorflow/privacy
$ cd privacy
$ pip install -e .
```

