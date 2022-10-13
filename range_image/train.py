import torch
from torch._C import device
from torch.utils.data import DataLoader
from range_dataloader import RangeDataset
from model import CNN_LSTM
from eval_model import EVAL_CNN_LSTM
from torch.autograd import Variable
import numpy as np
import os
from tensorboardX import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

writer = SummaryWriter(log_dir = "/home/mingxing/range_image/loss")

def eval_model(model, loader, print_info = False, plot = False, save = False):
    eval_iter = iter(loader)
    total_loss = 0
    groud_truth = []
    predict_vel = []
    for i, data in enumerate(eval_iter):
        feature, labels = data
        with torch.no_grad():
            loss_func = torch.nn.MSELoss(reduction='mean')
            prediction = model(feature.to(device))

            labels = labels.reshape(-1, 10, 1)
    
            loss = loss_func(prediction, torch.tensor(labels).to(device))
            total_loss += loss
    total_loss /= len(loader)
    return total_loss


def train():
    data_path = "/home/mingxing/NanyangLink"
    train_dataset = RangeDataset(path = data_path, Train = True, is_augmentation = False, is_seq = True, acc = 0.5)
    eval_dataset = RangeDataset(path = data_path, Train = True, is_augmentation = False, is_seq = True, acc = 0.5)

    train_dataloader = DataLoader(
                        train_dataset,
                        batch_size=64,
                        shuffle=True,
                        num_workers=16)

    eval_dataloader = DataLoader(
                    eval_dataset,
                    batch_size=64,
                    shuffle=False,
                    num_workers=16)

    model = CNN_LSTM()
    model.to(device)
    optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=0.0002,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    save = False
    best_loss = 100
    for epoch in range(100):
        train_iter = iter(train_dataloader)
        accum_loss, accum_iter = 0, 0
        tot_iter = 0
        # Training
        model.train()
        for i, data in enumerate(train_iter):
            feature, labels = data
            
            out = model(feature.to(device))
            optimizer.zero_grad()

            labels = labels.reshape(-1, 10, 1)

            loss = loss_fn(out, labels.to(torch.float32).to(device))
            loss.backward()
            optimizer.step()

            accum_loss += loss.item()
            accum_iter += 1
            tot_iter += 1

            if tot_iter % 10 == 0 or tot_iter == 1:
                print(
                    f'Iter: {tot_iter}, Epoch: {epoch}, Loss: {accum_loss / accum_iter}'
                )
                accum_loss, accum_iter = 0, 0
        eval_loss = eval_model(model.eval(), eval_dataloader, print_info = False, save = save)
        writer.add_scalar('loss', eval_loss, global_step=epoch)
        save_model = False
        if best_loss == None:
            best_loss = eval_loss
            save_model = True
        if eval_loss < best_loss:
            best_loss = eval_loss
            save_model = True

        if save_model:
            fn_pth = '%s-%.8f-%04d.pth' % ('check_point/CNN_LSTM', best_loss, epoch)
            torch.save(model.state_dict(), os.path.join("/home/mingxing/range_image", fn_pth))
            eval_loss = eval_model(model.eval(), eval_dataloader, print_info = False, save = save_model)

def eval_data():
    data_path = "/home/mingxing/NanyangLink"
    model = EVAL_CNN_LSTM()
    model.load_state_dict(torch.load("/home/mingxing/range_image/check_point/CNN_LSTM-0.00006027-0030.pth"))
    model.eval()
    eval_dataset = RangeDataset(path = data_path, Train = True, is_augmentation = False, is_seq = True, acc = 0.5)
    eval_dataloader = DataLoader(
                    eval_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=16)

    eval_iter = iter(eval_dataloader)
    total_loss = 0
    groud_truth = []
    predict_vel = []

    h = Variable(torch.zeros(2, 1, 128))
    c = Variable(torch.zeros(2, 1, 128))
    for i, data in enumerate(eval_iter):
        feature, labels = data
        with torch.no_grad():
            loss_func = torch.nn.MSELoss(reduction='mean')
            prediction, h, c = model(feature, h, c)

            prediction = prediction.reshape(-1)
            labels = labels.reshape(-1)

            loss = loss_func(prediction, torch.tensor(labels))
            total_loss += loss

            predict_vel.extend(prediction.numpy().tolist())
            groud_truth.extend(labels)

    np.save("truth.npy", groud_truth)
    np.save("predict_vel.npy", predict_vel)

if __name__ == '__main__':
    train()
    # eval_data()