# Inter-SubNet

This Git repository for the official PyTorch implementation of **"Inter-SubNet: Speech Enhancement with Subband Interaction"**,  accepted by ICASSP 2023. (**The paper will be released soon!**)

▶[[Demo](https://rookiejunchen.github.io/Inter-SubNet_demo/)] 💿[[Checkpoint](https://drive.google.com/file/d/1j9jdXRxPhXLE93XlYppCQtcOqMOJNjdt/view?usp=sharing)]



## Requirements

- Linux or macOS 

- python>=3.6

- Anaconda or Miniconda

- NVIDIA GPU + CUDA CuDNN (CPU can also be supported)



### Environment && Installation

Install Anaconda or Miniconda, and then install conda and pip packages:

```shell
# Create conda environment
conda create --name speech_enhance python=3.8
conda activate speech_enhance

# Install conda packages
# Check python=3.8, cudatoolkit=10.2, pytorch=1.7.1, torchaudio=0.7
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install tensorboard joblib matplotlib

# Install pip packages
# Check librosa=0.8
pip install Cython
pip install librosa pesq pypesq pystoi tqdm toml colorful mir_eval torch_complex

# (Optional) If you want to load "mp3" format audio in your dataset
conda install -c conda-forge ffmpeg
```



### Quick Usage

Clone the repository:

```shell
git clone https://github.com/RookieJunChen/Inter-SubNet.git
cd Inter-SubNet
```

Download the [pre-trained checkpoint](https://drive.google.com/file/d/1j9jdXRxPhXLE93XlYppCQtcOqMOJNjdt/view?usp=sharing), and input commands:

```shell
source activate speech_enhance
python -m speech_enhance.tools.inference \
  -C config/inference.toml \
  -M $MODEL_DIR \
  -I $INPUT_DIR \
  -O $OUTPUT_DIR
```

<br/> 

## Start Up

### Clone

```shell
git clone https://github.com/RookieJunChen/Inter-SubNet.git
cd Inter-SubNet
```



### Data preparation

#### Train data

Please prepare your data in the data dir as like:

- data/DNS-Challenge/DNS-Challenge-interspeech2020-master/
- data/DNS-Challenge/DNS-Challenge-master/

and set the train dir in the script `run.sh`.

Then:

```shell
source activate speech_enhance
bash run.sh 0   # peprare training list or meta file
```

#### Test data

Please prepare your test cases dir like: `data/test_cases_<name>`, and set the test dir in the script `run.sh`.



### Training

First, you need to modify the various configurations in `config/train.toml` for training.

Then you can run training:

```shell
source activate speech_enhance
bash run.sh 1   
```



### Inference

After training, you can enhance noisy speech.  Before inference, you first need to modify the configuration in `config/inference.toml`.

You can also run inference:

```shell
source activate speech_enhance
bash run.sh 2
```

Or you can just use `inference.sh`:

```shell
source activate speech_enhance
bash inference.sh
```





### Eval

Calculating bjective metrics (SI_SDR, STOI, WB_PESQ, NB_PESQ, etc.) :

```shell
bash metrics.sh
```

For test set without reference, you can obtain subjective scores (DNS_MOS and  NISQA, etc) through [DNSMOS](https://github.com/RookieJunChen/dns_mos_calculate) or [NISQA](https://github.com/RookieJunChen/my_NISQA).

