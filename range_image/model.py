import torch

class CNN_LSTM(torch.nn.Module):

    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=16,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,3,1,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,3,1,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,1,2,0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )

        self.rnn = torch.nn.LSTM(51200, 128, 2, batch_first=True)

        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(128,64)
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(64,1)
        )

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.conv1(c_in)
        c_out = self.conv2(c_out)
        c_out = self.conv3(c_out)
        c_out = self.conv4(c_out)
        print(C, H, W)
        r_in = c_out.view(batch_size, timesteps, -1)
        print(r_in.shape)
        r_out, _ = self.rnn(r_in)

        out = self.mlp1(r_out)
        out = self.mlp2(out)
        return out