CUDA_VISIBLE_DEVICES=0 \
cd .. \
python train.py \
--model 'cmslstm' \
--dataset 'taxibj' \
--data_root './data/Moving_MNIST' \
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
