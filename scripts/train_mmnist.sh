CUDA_VISIBLE_DEVICES=0 \
cd .. \
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
