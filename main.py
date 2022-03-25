import numpy as np
import torch
from torchvision import transforms
import torch.optim as optim
from model import Model
from Omniglot_Burda import Omniglot_Burda
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.current_device())

LEARNING_RATE = 3e-4
BATCH_SIZE = 32
CODE_SIZE = 100
MEMORY_SIZE = 32
EPISODE_SIZE = 32
PSEUDOINVERSE_APPROX_STEP = 7
DIRECT_WRITING = True
ORDERING = False

model = Model(input_size=784, memory_size=MEMORY_SIZE, code_size=CODE_SIZE,
            direct_writing=DIRECT_WRITING, ordering=ORDERING, pseudoinverse_approx_step=PSEUDOINVERSE_APPROX_STEP)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

trans = transforms.Compose([transforms.Normalize((0.5,), (0.5,)),
                            lambda x: x > 0,
                            lambda x: x.float()])

DATA_ROOT = 'chardata.mat'
trainset = Omniglot_Burda(root=DATA_ROOT, train=True, transform=trans)
testset = Omniglot_Burda(root=DATA_ROOT, train=False, transform=trans)

train_loader = torch.utils.data.DataLoader(
                  dataset=trainset,
                  batch_size=BATCH_SIZE * EPISODE_SIZE,
                  shuffle=True,)

if BATCH_SIZE * EPISODE_SIZE < 8000:
    test_loader = torch.utils.data.DataLoader(
                      dataset=testset,
                      batch_size=BATCH_SIZE * EPISODE_SIZE,
                      shuffle=True,)
else:
    test_loader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=8000,
        shuffle=True, )

iteration = 0
epoch = 0
kl_total_list = []
print('START TRAINING...')
while True:
    recon_loss_epoch = []
    kl_w_epoch = []
    kl_M_epoch = []
    bijective_epoch = []
    for batch_idx, (X, target) in enumerate(train_loader):
        X = X.view(X.shape[0], -1).cuda()

        if X.shape[0] == BATCH_SIZE * EPISODE_SIZE:
            input_recon, (recon_loss, dkl_w, dkl_M, bijective_loss) = model(input=X, episode_size=EPISODE_SIZE)
            torch.autograd.set_detect_anomaly(True)

            total_loss = recon_loss + dkl_w + bijective_loss

            recon_loss_epoch.append(recon_loss.item())
            kl_w_epoch.append(dkl_w.item())
            kl_M_epoch.append(dkl_M.item())
            bijective_epoch.append(bijective_loss.item())

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

    epoch += 1
    print(epoch, np.mean(recon_loss_epoch), np.mean(kl_w_epoch), np.mean(kl_M_epoch), np.mean(bijective_epoch))

    if epoch % 10 == 0:
        test_loss = 0
        for batch_idx, (X, target) in enumerate(test_loader):
            X = X.view(X.shape[0], -1).cuda()
            if BATCH_SIZE * EPISODE_SIZE < 8000:
                if X.shape[0] == BATCH_SIZE * EPISODE_SIZE:
                    input_recon, (recon_loss, dkl_w, dkl_M, bijective_loss) = model(input=X, episode_size=EPISODE_SIZE)
                    test_loss += torch.mean(recon_loss).item() + torch.mean(dkl_w).item()
                    del input_recon, recon_loss, dkl_w, dkl_M, bijective_loss
            else:
                if X.shape[0] == 8000:
                    input_recon, (recon_loss, dkl_w, dkl_M, bijective_loss) = model(input=X, episode_size=8000)
                    test_loss += torch.mean(recon_loss).item() + torch.mean(dkl_w).item()
                    del input_recon, recon_loss, dkl_w, dkl_M, bijective_loss
        print('test loss:', test_loss / batch_idx)

    if epoch >= 1000:
        break
