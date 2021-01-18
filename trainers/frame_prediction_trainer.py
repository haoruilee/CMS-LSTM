import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import cv2
from tqdm import tqdm
from models import get_convrnn_model
import utils
from tensorboardX import SummaryWriter


class MSEL1Loss(nn.Module):
    def __init__(self, size_average=False, reduce=True, alpha=1.0):
        super(MSEL1Loss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.alpha = alpha

        self.mse_criterion = nn.MSELoss(size_average=self.size_average, reduce=self.reduce)
        self.l1_criterion = nn.L1Loss(size_average=self.size_average, reduce=self.reduce)

    def __call__(self, input, target):
        mse_loss = self.mse_criterion(input, target)
        l1_loss = self.l1_criterion(input, target)
        loss = mse_loss + self.alpha * l1_loss
        return loss / 2.


class FPTrainer():
    def __init__(self, opt):
        self.opt = opt
        self.save_dir = opt.log_dir
        self.dataset = opt.dataset
        self.batch_size = opt.batch_size
        self.patch_size = opt.patch_size

        self.dtype = torch.cuda.FloatTensor

        self.start_epoch = 0
        self.eta = 1.0
        self.total_iter = 0
        num_iters = int(opt.total_epoch * opt.epoch_size / 2)
        self.delta = float(1) / num_iters

        self.seq_len = opt.seq_len
        self.pre_len = opt.pre_len
        self.eval_len = opt.eval_len

        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc

        self.epoch_size = opt.epoch_size

        self.shape = [int(opt.image_width / opt.patch_size), int(opt.image_height / opt.patch_size)]

        self.rnn_size = opt.rnn_size
        self.rnn_nlayer = opt.rnn_nlayer

        self.filter_size = opt.filter_size

        ic = self.input_nc * opt.patch_size ** 2
        oc = self.output_nc * opt.patch_size ** 2

        # tensorboard
        # ---------------- visualization with tensorboardX ----------
        train_log_dir = os.path.join(self.save_dir, 'runs/train')
        if not os.path.exists(train_log_dir):
            os.mkdir(train_log_dir)
        self.writer_train = SummaryWriter(log_dir=train_log_dir)

        test_log_dir = os.path.join(self.save_dir, 'runs/test')
        if not os.path.exists(test_log_dir):
            os.mkdir(test_log_dir)
        self.writer_test = SummaryWriter(log_dir=test_log_dir)

        # setting dataset
        train_data, valid_data, test_data = utils.load_dataset(opt)

        self.train_loader = DataLoader(train_data,
                                       num_workers=opt.data_threads,
                                       batch_size=opt.batch_size,
                                       shuffle=True,
                                       drop_last=True,
                                       pin_memory=True)

        self.test_loader = DataLoader(test_data,
                                      num_workers=opt.data_threads,
                                      batch_size=opt.batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      pin_memory=True)

        self.valid_loader = DataLoader(valid_data,
                                       num_workers=opt.data_threads,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       drop_last=False,
                                       pin_memory=True)

        def get_training_batch():
            while True:
                for sequence in self.train_loader:
                    batch = utils.normalize_data(opt, self.dtype, sequence)
                    yield batch

        self.training_batch_generator = get_training_batch()

        def get_testing_batch():
            while True:
                for sequence in self.test_loader:
                    batch = utils.normalize_data(opt, self.dtype, sequence)
                    yield batch

        self.testing_batch_generator = get_testing_batch()

        # set model
        self.model = get_convrnn_model(opt.model, input_chans=ic,
                                       output_chans=oc,
                                       hidden_size=self.rnn_size,
                                       filter_size=self.filter_size,
                                       num_layers=self.rnn_nlayer,
                                       img_size=opt.image_height // opt.patch_size)
        self.model.cuda()

        # set optimizer
        if opt.optimizer == 'adam':
            optimizer = optim.Adam
        elif opt.optimizer == 'rmsprop':
            optimizer = optim.RMSprop
        elif opt.optimizer == 'sgd':
            optimizer = optim.SGD
        elif opt.optimizer == 'adamw':
            optimizer = optim.AdamW
        else:
            raise ValueError('Unknown optimizer: %s' % opt.optimizer)

        self.optimizer = optimizer(self.model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        self.scheduler = utils.get_scheduler(self.optimizer, self.opt, (opt.total_epoch - self.start_epoch))

        # load model
        if opt.resume:
            if not os.path.isfile(opt.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.eta = checkpoint['eta']
            self.total_iter = self.start_epoch * self.epoch_size
            print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch']))

        # criterion
        if opt.criterion == "L1":
            self.criterion = torch.nn.L1Loss(size_average=False, reduce=True).cuda()
            print('The criterion type is L1!')
        elif opt.criterion == "BCE":
            self.criterion = torch.nn.BCELoss(size_average=False, reduce=True).cuda()
            print('The criterion type is BCE!')
        elif opt.criterion == "MSE&L1":
            self.criterion = MSEL1Loss(size_average=False, reduce=True, alpha=1.0).cuda()
            print('The criterion type is MSE + L1!')
        else:
            self.criterion = torch.nn.MSELoss(size_average=False, reduce=True).cuda()
            print('The criterion type is MSE!')

        # Print networks
        print('---------- Networks initialized -------------')
        utils.print_network(self.model, opt.model)

    def name(self):
        return 'Frame Prediction Trainer'

    def set_input(self, input):
        # X: len, batchsize, inchains, size(0), size(1)
        if self.patch_size > 1:
            x = [utils.reshape_patch(img, self.patch_size) for img in input]
        else:
            x = input

        reverse_x = x[::-1]

        random_flip = np.random.random_sample(
            (self.pre_len - 1, self.batch_size))
        true_token = (random_flip < self.eta)
        one = torch.FloatTensor(1, x[0].size(1), x[0].size(2), x[0].size(3)).fill_(1.0).cuda()
        zero = torch.FloatTensor(1, x[0].size(1), x[0].size(2), x[0].size(3)).fill_(0.0).cuda()

        masks = []
        for t in range(self.pre_len - 1):
            masks_b = []
            for i in range(self.batch_size):
                if true_token[t, i]:
                    masks_b.append(one)
                else:
                    masks_b.append(zero)
            mask = torch.cat(masks_b, 0)  # along batchsize
            masks.append(mask)
        return x, reverse_x, masks

    def forward(self, x, mask):
        gen_ims = []
        x_gen = self.model(x[0], init_hidden=True)

        for t in range(1, self.seq_len + self.pre_len - 1):
            if t < self.seq_len:
                inputs = x[t]
            else:
                inputs = mask[t - self.seq_len] * x[t] + (1 - mask[t - self.seq_len]) * x_gen

            x_gen = self.model(inputs, init_hidden=False)

            gen_ims.append(x_gen)

        gen_ims = torch.stack(gen_ims, 1)
        images = torch.stack(x[2:], 1)

        loss = self.criterion(gen_ims, images)
        loss /= 2.0
        return loss

    def save_checkpoint(self, checkpoint, network_label, epoch_label):
        save_filename = '%s_%s_net_%s.pth.tar' % (self.dataset, network_label, epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(checkpoint, save_path)

    def train_epoch(self, epoch):
        self.model.train()
        # epoch_rec = 0.0
        info_dict = {'loss': 0.0}
        describe = '[' + self.opt.model + ',' + self.opt.dataset + ']:' + 'Epoch ' + str(epoch)
        pbar = tqdm(total=self.epoch_size, desc=describe)

        for i in range(self.epoch_size):
            x = next(self.training_batch_generator)
            loss = self.optimize_parameters(x)
            # epoch_rec += loss

            with open(os.path.join(self.save_dir, 'train_loss_%s_%s.txt' % (self.opt.model, self.opt.dataset)),
                      mode='a') as f:
                f.write('%0.8f \n' % (loss))

            self.total_iter += 1
            self.writer_train.add_scalar('Train/loss', loss, self.total_iter)
            self.writer_train.add_scalar('Train/Eta', self.eta, self.total_iter)

            self.eta -= self.delta
            self.eta = max(self.eta, 0.0)

            info_dict['loss'] = loss
            pbar.set_postfix(info_dict)
            pbar.update(1)
        pbar.close()

        # save epoch
        self.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'eta': self.eta,
        }, self.opt.model, 'last')

        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]
        print('learning rate = %.7f' % lr)

    def test(self, epoch):
        self.model.eval()
        rec = 0

        result_path = os.path.join(self.save_dir, 'results', str(epoch))
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        psnr = np.zeros(self.eval_len)
        ssim = np.zeros(self.eval_len)
        mae = np.zeros(self.eval_len)
        sharp = np.zeros(self.eval_len)
        mse = np.zeros(self.eval_len)

        index = 0
        total_index = 0

        describe = '[Testing]:Epoch ' + str(epoch)
        pbar = tqdm(total=len(self.test_loader), desc=describe)

        for batch in self.test_loader:
            # for batch in self.test_loader:
            x = utils.normalize_data(self.opt, self.dtype, batch)
            rec += self.evaluation(x)
            index += 1
            total_index += x[0].size(0)  # bs

            gt = []
            pred = []
            for i in range(self.eval_len):
                x1 = x[i + self.seq_len].data.cpu().numpy()
                x2 = self.preds[i + self.seq_len].data.cpu().numpy()
                gt.append(x1)
                pred.append(x2)

            mse_, mae_, ssim_, psnr_, sharp_ = utils.eval_seq_batch(gt, pred)
            mse += mse_
            mae += mae_
            ssim += ssim_
            psnr += psnr_
            sharp += sharp_

            if index < 11:
                path = os.path.join(result_path, str(index))
                if not os.path.exists(path):
                    os.mkdir(path)
                for i in range(self.seq_len + self.eval_len):
                    name = 'gt' + str(i + 1) + '.png'
                    file_name = os.path.join(path, name)
                    img_gt = x[i][0].data.cpu()
                    img_gt = img_gt.transpose(0, 1).transpose(1, 2).numpy()
                    img_gt = np.uint8(img_gt * 255)

                    if 2 in img_gt.shape:
                        cv2.imwrite(file_name, img_gt[:, :, :1])
                        continue
                    cv2.imwrite(file_name, img_gt)

                for i in range(self.eval_len):
                    name = 'pd' + str(i + self.seq_len + 1) + '.png'
                    file_name = os.path.join(path, name)
                    img_pd = self.preds[i + self.seq_len][0].data.cpu()
                    img_pd = img_pd.transpose(0, 1).transpose(1, 2).clamp(0, 1).numpy()
                    img_pd = np.uint8(img_pd * 255)

                    if 2 in img_pd.shape:
                        cv2.imwrite(file_name, img_pd[:, :, :1])
                        continue

                    cv2.imwrite(file_name, img_pd)

            if index == 1:
                gt = torch.stack(x, dim=1)  # B, T, C, H, W
                pd = torch.stack(self.preds, dim=1)
                gif = torch.cat([gt, pd], dim=0)  # cat along batch
                gif = gif.data.cpu().clamp(0, 1)
                self.writer_test.add_video('Test/gt&pred', gif, epoch)

            pbar.update(1)
        pbar.close()

        rec = rec / index
        mse /= total_index
        mae /= total_index
        ssim /= total_index
        psnr /= total_index
        sharp /= total_index

        # ----------- log the frame-wise measurement
        with open(os.path.join(self.save_dir, 'test_result_%s.txt' % (self.dataset)), mode='a') as f:
            f.write('####################### frame-wise results at epoch: %04d ####################### \n' % (epoch))

            f.write('- mse: mean %04f -' % (np.mean(mse)))
            for t in range(self.eval_len):
                f.write('-[%d: %04f]-' % (t + self.pre_len, mse[t]))
            f.write('\n')

            f.write('- mae: mean %04f -' % (np.mean(mae)))
            for t in range(self.eval_len):
                f.write('[%d: %04f] ' % (t + self.pre_len, mae[t]))
            f.write('\n')

            f.write('- ssim: mean %04f -' % (np.mean(ssim)))
            for t in range(self.eval_len):
                f.write('[%d: %04f] ' % (t + self.pre_len, ssim[t]))
            f.write('\n')

            f.write('- psnr: mean %04f -' % (np.mean(psnr)))
            for t in range(self.eval_len):
                f.write('[%d: %04f] ' % (t + self.pre_len, psnr[t]))
            f.write('\n')

            f.write('- sharp: mean %04f -' % (np.mean(sharp)))
            for t in range(self.eval_len):
                f.write('[%d: %04f] ' % (t + self.pre_len, sharp[t]))
            f.write('\n')

        self.writer_test.add_scalar('Test/rec_loss', rec, epoch)
        self.writer_test.add_scalar('Test/mse', np.mean(mse), epoch)
        self.writer_test.add_scalar('Test/mae', np.mean(mae), epoch)
        self.writer_test.add_scalar('Test/ssim', np.mean(ssim), epoch)
        self.writer_test.add_scalar('Test/psnr', np.mean(psnr), epoch)
        self.writer_test.add_scalar('Test/sharp', np.mean(sharp), epoch)

    def evaluation(self, x):
        # patch
        if self.patch_size > 1:
            x = [utils.reshape_patch(img, self.patch_size) for img in x]
        # --------------------------------------------------------------
        loss = 0.0
        self.preds = [x[0], x[1]]
        x_gen = self.model(x[0], init_hidden=True)
        gen_ims = []

        for t in range(1, self.seq_len + self.eval_len - 1):
            if t < self.seq_len:
                inputs = x[t]
            else:
                inputs = x_gen

            x_gen = self.model(inputs, init_hidden=False)
            gen_ims.append(x_gen)

            # -----------------------------------------------------------
            if t < self.seq_len - 1:
                self.preds.append(x[t + 1])
            else:
                self.preds.append(x_gen)

        # unpatch
        if self.patch_size > 1:
            self.preds = [utils.reshape_patch_back(img, self.patch_size) for img in self.preds]

        gen_ims = torch.stack(gen_ims, 1)
        images = torch.stack(x[2:], 1)
        loss = self.criterion(gen_ims, images)
        loss /= 2.0  # for consistency with tensorflow

        return loss.detach().to("cpu").item() / (self.batch_size * (self.seq_len + self.eval_len - 2))

    def optimize_parameters(self, x):
        x, x_rev, mask = self.set_input(x)

        self.optimizer.zero_grad()
        self.loss_1 = self.forward(x, mask)

        self.loss_1.backward()
        self.optimizer.step()

        self.optimizer.zero_grad()
        self.loss_2 = self.forward(x_rev, mask)

        self.loss_2.backward()
        self.optimizer.step()

        loss = (self.loss_1 + self.loss_2) / 2
        return loss.detach().to("cpu").item()

    def finish(self):
        self.preds = []
        self.writer_train.export_scalars_to_json(os.path.join(self.save_dir, 'runs', 'train_all_scalars.json'))
        self.writer_train.close()
        self.writer_test.export_scalars_to_json(os.path.join(self.save_dir, 'runs', 'test_all_scalars.json'))
        self.writer_test.close()
