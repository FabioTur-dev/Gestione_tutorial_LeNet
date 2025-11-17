# 🧠 MNIST LeNet Project

A minimal yet complete **Python deep learning project** for image classification on **MNIST**.

It implements a **LeNet-style convolutional neural network**, uses the **GPU when available**, and follows clean
**software engineering practices**:

- Clear project structure
- YAML-based configuration
- Reproducible experiments (fixed random seeds)
- Separate training and evaluation scripts
- Ready to be extended by students

## 📁 Project Structure

```text
mnist-lenet-project/
├─ README.md
├─ requirements.txt
├─ configs/
│  └─ dev.yaml
├─ data/
│  ├─ raw/
│  └─ processed/
├─ artifacts/
├─ outputs/
└─ src/
   ├─ common/
   │  ├─ config.py
   │  └─ __init__.py
   └─ models/
      ├─ lenet.py
      ├─ train.py
      ├─ evaluate.py
      └─ __init__.py
```

## ⚙️ Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## 🚀 Train

```bash
python -m src.models.train --config configs/dev.yaml --seed 42
```

## 📊 Evaluate

```bash
python -m src.models.evaluate --config configs/dev.yaml --seed 42
```
