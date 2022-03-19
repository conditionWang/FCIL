from GLFC import GLFC_model
from ResNet import resnet18_cbam
import torch
import copy
import random
import os.path as osp
import os
from myNetwork import network, LeNet
from Fed_utils import * 
from ProxyServer import * 
from mini_imagenet import *
from tiny_imagenet import *
from option import args_parser

args = args_parser()

## parameters for learning
feature_extractor = resnet18_cbam()
num_clients = args.num_clients
old_client_0 = []
old_client_1 = [i for i in range(args.num_clients)]
new_client = []
models = []

## seed settings
setup_seed(args.seed)

## model settings
model_g = network(args.numclass, feature_extractor)
model_g = model_to_device(model_g, False, args.device)
model_old = None

train_transform = transforms.Compose([transforms.RandomCrop((args.img_size, args.img_size), padding=4),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ColorJitter(brightness=0.24705882352941178),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
test_transform = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(), 
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

if args.dataset == 'cifar100':
    train_dataset = iCIFAR100('dataset', transform=train_transform, download=True)
    test_dataset = iCIFAR100('dataset', test_transform=test_transform, train=False, download=True)

elif args.dataset == 'tiny_imagenet':
    train_dataset = Tiny_Imagenet('./tiny-imagenet-200', train_transform=train_transform, test_transform=test_transform)
    train_dataset.get_data()
    test_dataset = train_dataset

else:
    train_dataset = Mini_Imagenet('./train', train_transform=train_transform, test_transform=test_transform)
    train_dataset.get_data()
    test_dataset = train_dataset

encode_model = LeNet(num_classes=100)
encode_model.apply(weights_init)

for i in range(125):
    model_temp = GLFC_model(args.numclass, feature_extractor, args.batch_size, args.task_size, args.memory_size,
                 args.epochs_local, args.learning_rate, train_dataset, args.device, encode_model)
    models.append(model_temp)

## the proxy server
proxy_server = proxyServer(args.device, args.learning_rate, args.numclass, feature_extractor, encode_model, train_transform)

## training log
output_dir = osp.join('./training_log', args.method, 'seed' + str(args.seed))
if not osp.exists(output_dir):
    os.system('mkdir -p ' + output_dir)
if not osp.exists(output_dir):
    os.mkdir(output_dir)

out_file = open(osp.join(output_dir, 'log_tar_' + str(args.task_size) + '.txt'), 'w')
log_str = 'method_{}, task_size_{}, learning_rate_{}'.format(args.method, args.task_size, args.learning_rate)
out_file.write(log_str + '\n')
out_file.flush()

classes_learned = args.task_size
old_task_id = -1
for ep_g in range(args.epochs_global):
    pool_grad = []
    model_old = proxy_server.model_back()
    task_id = ep_g // args.tasks_global

    if task_id != old_task_id and old_task_id != -1:
        overall_client = len(old_client_0) + len(old_client_1) + len(new_client)
        new_client = [i for i in range(overall_client, overall_client + args.task_size)]
        old_client_1 = random.sample([i for i in range(overall_client)], int(overall_client * 0.9))
        old_client_0 = [i for i in range(overall_client) if i not in old_client_1]
        num_clients = len(new_client) + len(old_client_1) + len(old_client_0)
        print(old_client_0)

    if task_id != old_task_id and old_task_id != -1:
        classes_learned += args.task_size
        model_g.Incremental_learning(classes_learned)
        model_g = model_to_device(model_g, False, args.device)
    
    print('federated global round: {}, task_id: {}'.format(ep_g, task_id))

    w_local = []
    clients_index = random.sample(range(num_clients), args.local_clients)
    print('select part of clients to conduct local training')
    print(clients_index)

    for c in clients_index:
        local_model, proto_grad = local_train(models, c, model_g, task_id, model_old, ep_g, old_client_0)
        w_local.append(local_model)
        if proto_grad != None:
            for grad_i in proto_grad:
                pool_grad.append(grad_i)

    ## every participant save their current training data as exemplar set
    print('every participant start updating their exemplar set and old model...')
    participant_exemplar_storing(models, num_clients, model_g, old_client_0, task_id, clients_index)
    print('updating finishes')

    print('federated aggregation...')
    w_g_new = FedAvg(w_local)
    w_g_last = copy.deepcopy(model_g.state_dict())
    
    model_g.load_state_dict(w_g_new)

    proxy_server.model = copy.deepcopy(model_g)
    proxy_server.dataloader(pool_grad)

    acc_global = model_global_eval(model_g, test_dataset, task_id, args.task_size, args.device)
    log_str = 'Task: {}, Round: {} Accuracy = {:.2f}%'.format(task_id, ep_g, acc_global)
    out_file.write(log_str + '\n')
    out_file.flush()
    print('classification accuracy of global model at round %d: %.3f \n' % (ep_g, acc_global))

    old_task_id = task_id
