# HiFi-GAN

My implementation of "HiFi-GAN: Generative Adversarial Networks
for Efficient and High Fidelity Speech Synthesis" paper

[Paper HiFi-GAN](https://arxiv.org/pdf/2010.05646.pdf)

[All runs](https://wandb.ai/messlav/Hi-Fi-GAN)

[Wandb report](https://wandb.ai/messlav/Hi-Fi-GAN/reports/HiFi-GAN--VmlldzozMTk3MjQy?accessToken=jiszk8sq5mh1ly8pzfm3anu7pvffhx2lnd8rlp5f0piw4hgm6bhjm1ka8xo3e0bt)


# Reproduce code

1. Clone repo and install packages

```python
git clone https://github.com/messlav/HiFi-GAN
cd HiFi-GAN
pip install -r requirements.txt
```

2. Get data

```python
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2
```

3. Train GAN

```python
python3 train.py
```

4. Download weights

```python
import gdown

gdown.download('https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx')
```
