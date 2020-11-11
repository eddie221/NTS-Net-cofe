import os
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from config import BATCH_SIZE, PROPOSAL_NUM, SAVE_FREQ, LR, WD, resume, save_dir, EPOCH, INPUT_SIZE
from core import model
import torchvision.transforms as transforms
from PIL import Image
import torchvision

from core.utils import init_log, progress_bar

data_dir = '../skin/ISIC 2019/'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
start_epoch = 1
save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
if os.path.exists(save_dir):
    raise NameError('model dir exists!')
os.makedirs(save_dir)
logging = init_log(save_dir)
_print = logging.info

data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((600, 600), Image.BILINEAR),
            transforms.RandomCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            
        ]),
        'val': transforms.Compose([
            transforms.Resize((600, 600), Image.BILINEAR),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
}

# read dataset
image_datasets = {x : torchvision.datasets.ImageFolder(os.path.join(data_dir, x),
                                                           data_transforms[x]) 
                        for x in ['train', 'val']}
    
image_dataloader = {x : torch.utils.data.DataLoader(image_datasets[x],
                                                    batch_size=BATCH_SIZE,
                                                    #sampler = data_sampler[x],
                                                    shuffle=True,
                                                    num_workers=16)
                    for x in ['train', 'val']}
# =============================================================================
# trainset = dataset.CUB(root='./CUB_200_2011', is_train=True, data_len=None)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
#                                           shuffle=True, num_workers=8, drop_last=False)
# testset = dataset.CUB(root='./CUB_200_2011', is_train=False, data_len=None)
# testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
#                                          shuffle=False, num_workers=8, drop_last=False)
# =============================================================================
# define model
net = model.attention_net(topN=PROPOSAL_NUM)
if resume:
    ckpt = torch.load(resume)
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1
creterion = torch.nn.CrossEntropyLoss()

# define optimizers
raw_parameters = list(net.pretrained_model.parameters())
part_parameters = list(net.proposal_net.parameters())
concat_parameters = list(net.concat_net.parameters())
partcls_parameters = list(net.partcls_net.parameters())

raw_optimizer = torch.optim.SGD(raw_parameters, lr=LR, momentum=0.9, weight_decay=WD)
concat_optimizer = torch.optim.SGD(concat_parameters, lr=LR, momentum=0.9, weight_decay=WD)
part_optimizer = torch.optim.SGD(part_parameters, lr=LR, momentum=0.9, weight_decay=WD)
partcls_optimizer = torch.optim.SGD(partcls_parameters, lr=LR, momentum=0.9, weight_decay=WD)
schedulers = [MultiStepLR(raw_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(concat_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(part_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(partcls_optimizer, milestones=[60, 100], gamma=0.1)]
net = net.cuda()
net = DataParallel(net)
VAL_MAX_ACC = 0
for epoch in range(EPOCH):
    print("save_dir : {}".format(save_dir))
    # begin training
    _print('--' * 50)
    net.train()
    train_loss = 0
    train_correct = 0
    test_loss = 0
    test_correct = 0
    for phase in ['train', 'val']:
        total = 0
        if phase == 'train':
            net.train(True)
        else:
            net.train(False)
            
        for i, data in enumerate(image_dataloader[phase]):
            img, label = data[0].cuda(), data[1].cuda()
            batch_size = img.size(0)
            raw_optimizer.zero_grad()
            part_optimizer.zero_grad()
            concat_optimizer.zero_grad()
            partcls_optimizer.zero_grad()
            cam_label = torch.tensor([[0, 1]]).repeat(img.shape[0], 1).unsqueeze(2).unsqueeze(3).cuda()
            
    
            raw_logits, concat_logits, part_logits, _, top_n_prob, cam, cam_rf = net(img)
            part_loss = model.list_loss(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                        label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)).view(batch_size, PROPOSAL_NUM)
            raw_loss = creterion(raw_logits, label)
            concat_loss = creterion(concat_logits, label)
            rank_loss = model.ranking_loss(top_n_prob, part_loss)
            partcls_loss = creterion(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                     label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))
            er_loss = torch.mean(torch.abs(cam[:, 1:, :, :] - cam_rf[:, 1:, :, :]))
    
            total_loss = raw_loss + rank_loss + concat_loss + partcls_loss + er_loss
            
            total_loss.backward()
            raw_optimizer.step()
            part_optimizer.step()
            concat_optimizer.step()
            partcls_optimizer.step()
            #progress_bar(i, len(trainloader), 'train')
            
            # calculate accuracy
            if phase == 'train': 
                _, concat_predict = torch.max(concat_logits, 1)
                total += batch_size
                train_correct += torch.sum(concat_predict.data == label.data)
                train_loss += concat_loss.item() * batch_size
            else:
                _, concat_predict = torch.max(concat_logits, 1)
                total += batch_size
                test_correct += torch.sum(concat_predict.data == label.data)
                test_loss += concat_loss.item() * batch_size
                
            progress_bar(i, len(image_dataloader[phase]), 'eval {} set'.format(phase))
        
        if phase == 'train':
            train_acc = float(train_correct) / total
            train_loss = train_loss / total
        else:
            test_acc = float(test_correct) / total
            test_loss = test_loss / total
    
        _print(
            'epoch:{} - {} loss: {:.6f} and train acc: {:.6f} total sample: {}'.format(
                epoch,
                phase,
                train_loss,
                train_acc,
                total))
        
        for scheduler in schedulers:
            scheduler.step()
    
        # save model
        if test_acc > VAL_MAX_ACC or epoch == EPOCH:
            VAL_MAX_ACC = test_acc
            net_state_dict = net.module.state_dict()
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'net_state_dict': net_state_dict},
                os.path.join(save_dir, '{}_{:.3f}.ckpt'.format(epoch, test_acc)))

print('finishing training')
