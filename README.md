# CMS-LSTM: Context-Embedding and Multi-Scale Spatiotemporal-Expression LSTM for Video Prediction

## Abstract
Extracting variation and spatiotemporal features via limited frames remains as an unsolved and challenging problem in video prediction. Inherent uncertainty among consecutive frames exacerbates the difficulty in long-term prediction. To tackle the problem, we focus on capturing context correlations and multi-scale spatiotemporal flows, then propose CMS-LSTM by integrating two effective and lightweight blocks, namely Context-Embedding (CE) and Spatiotemporal-Expression (SE) block, into ConvLSTM backbone. CE block is designed for abundant context interactions, while SE block focuses on multi-scale spatiotemporal expression in hidden states. The newly introduced blocks also facilitate other spatiotemporal models (e.g., PredRNN, SA-ConvLSTM) to produce representative implicit features for video prediction. Qualitative and quantitative experiments demonstrate the effectiveness and flexibility of our proposed method. We use fewer parameters to reach markedly state-of-the-art results on Moving MNIST and TaxiBJ datasets in numbers of metrics. All source code is available at https://github.com/czh-98/CMS-LSTM.


## Results

<div>
    <table width="600" border="0px">
      <tr>
        <th width="100">(Real)</th>
        <th width="100">Ours</th>
        <th width="100">PredRNN</th>
        <th width="100">PredRNN++</th>
        <th width="100">MIM</th>
        <th width="100">SA-LSTM</th>
      </tr>
    </table>
    <img height="100" width="100" src="results/gt1.gif">
    <img height="100" width="100" src="results/CMS-LSTM_pd1.gif">
    <img height="100" width="100" src="results/PredRNN_pd1.gif">
    <img height="100" width="100" src="results/PredRNN++_pd1.gif">
    <img height="100" width="100" src="results/MIM_pd1.gif">
    <img height="100" width="100" src="results/SA-ConvLSTM_pd1.gif">
</div>
<div>
    <img height="100" width="100" src="results/gt8.gif">
    <img height="100" width="100" src="results/CMS-LSTM_pd8.gif">
    <img height="100" width="100" src="results/PredRNN_pd8.gif">
    <img height="100" width="100" src="results/PredRNN++_pd8.gif">
    <img height="100" width="100" src="results/MIM_pd8.gif">
    <img height="100" width="100" src="results/SA-ConvLSTM_pd8.gif">
</div>


## Datasets
<!-- ## Pre-trained Models and Datasets -->
<!-- Pretrained Model will be released soon! -->
Moving MNIST and TaxiBJ datasets are avilliable at [here](https://drive.google.com/drive/folders/1Dl0WcevBRSsLn6KYJ7-zxMjLqo1S7WVr?usp=sharing):

## Setup
All code is developed and tested on a Nvidia RTX2080Ti the following environment.
- Ubuntu18.04.1
- CUDA 10.0
- cuDNN 7.6.4
- Python 3.8.5
- h5py 2.10.0
- imageio 2.9.0
- numpy 1.19.4
- opencv-python 4.4.0.46
- pandas 1.1.5
- pillow 8.0.1
- scikit-image 0.17.2
- scipy 1.5.4
- torch 1.7.1
- torchvision 0.8.2
- tqdm 4.51.0
## Code Architecture
```
CMS-LSTM.
©¦  README.md
©¦  test.py
©¦  train.py
©¦  utils.py
©¦
©À©¤data
©¦  ©¦  movingmnist.py
©¦  ©¸©¤ taxibj.py
©¦
©À©¤logs
©À©¤models
©¦  ©¦  basic_blocks.py
©¦  ©¦  CMSLSTM.py
©¦  ©¸©¤ __init__.py
©¦
©¦
©À©¤scripts
©¦      train_mmnist.sh
©¦      train_taxibj.sh
©¦
©¸©¤trainers
   ©¦  frame_prediction_trainer.py
   ©¸©¤ __init__.py

```

## Train
### Train in Moving MNIST
Use the `scripts/train_mmnist.sh` script to train the model. To train the default model on Moving MNIST simply use:
```shell
cd scripts
sh train_mmnist.sh
```
You might want to change the `--data_root` which point to paths on your system of the data.

```
CUDA_VISIBLE_DEVICES=0 \
python train.py \
--model 'cmslstm' \
--dataset 'mmnist' \
--data_root './data/Moving_MNIST' \
--lr 0.001 \
--batch_size 8 \
--epoch_size 200 \
--input_nc 1 \
--output_nc 1 \
--load_size 720 \
--image_width 64 \
--image_height 64 \
--patch_size 4 \
--rnn_size 64 \
--rnn_nlayer 4 \
--filter_size 3 \
--seq_len 10 \
--pre_len 10 \
--eval_len 10 \
--criterion 'MSE' \
--lr_policy 'cosine' \
--niter 5 \
--total_epoch 2000 \
--data_threads 4 \
--optimizer adamw
```
### Train in TaxiBJ
Use the `scripts/train_taxibj.sh` script to train the model. To train the default model on Moving MNIST simply use:
```shell
cd scripts
sh train_taxibj.sh
```
You might want to change the `--data_root` which point to paths on your system of the data.

```
CUDA_VISIBLE_DEVICES=0 \
cd .. \
python train.py \
--model 'cmslstm' \
--dataset 'taxibj' \
--data_root './data/TaxiBJ' \
--lr 0.001 \
--batch_size 8 \
--epoch_size 200 \
--input_nc 1 \
--output_nc 1 \
--load_size 720 \
--image_width 32 \
--image_height 32 \
--patch_size 2 \
--rnn_size 64 \
--rnn_nlayer 4 \
--filter_size 3 \
--seq_len 4 \
--pre_len 4 \
--eval_len 4 \
--criterion 'MSE&L1' \
--lr_policy 'cosine' \
--niter 5 \
--total_epoch 400 \
--data_threads 4 \
--optimizer adamw
```
<!-- ## BibTeX
TODO -->

## Questions
If you have any questions or problems regarding the code or paper do not hesitate to contact us.