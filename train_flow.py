################################################################################

flow_wrapper = NICEWrapper()
optim_flow = Adam(flow_wrapper.model.parameters(), lr=LR_FLOW, betas=(BETA1, BETA2))
fsvdd_loss = FSVDDLoss()

if cuda_state:
    flow_wrapper.model.cuda()
    fsvdd_loss.cuda()

fsvdd_loss.init_vars(img_map_wrapper.model, flow_wrapper.model, data_train.dataloader_train_flow)

################################################################################

losses_flow = []

print('Starting flow training...')
flow_wrapper.model.train()

for epoch in range(1, EPOCHS_FLOW + 1):  # First epoch id is 1, not 0

    loss_epoch = 0.
    outputs = torch.zeros((NB_IMGS_TRAIN_NI, INPUT_DIM))
    if cuda_state:
        outputs = outputs.cuda()

    pbar = tqdm(enumerate(data_train.dataloader_train_flow))

    for batch_idx, (data, _, idxes) in pbar:

        if cuda_state:
            data = data.cuda()

        optim_flow.zero_grad()
        with torch.no_grad():
            mapping = img_map_wrapper.model.forward(data, False)
        output = flow_wrapper.model(mapping)
        loss = fsvdd_loss(flow_wrapper.model, output)
        loss.backward()
        optim_flow.step()

        fsvdd_loss.update_W(flow_wrapper.model)

        loss_epoch += loss.data.cpu().numpy()
        outputs[idxes] = output.data.clone()

        if batch_idx % LOG_INTERVAL == 0:
            pbar.set_description('Train Epoch: {}/{} (lr: {:.2e}) [{}/{} ({:.0f}%)]. Loss: {:.3f} '.format(epoch,
                                                                                                           EPOCHS_FLOW,
                                                                                                           optim_flow.param_groups[0]['lr'],
                                                                                                           batch_idx * len(data),
                                                                                                           NB_IMGS_TRAIN_NI,
                                                                                                           100. * batch_idx / len(data_train.dataloader_train_flow),
                                                                                                           (loss_epoch / (batch_idx + 1)).item()))

        if batch_idx + 1 >= len(data_train.dataloader_train_flow):
            pbar.set_description('Train Epoch: {}/{} (lr: {:.2e}) [{}/{} ({:.0f}%)]. Loss: {:.3f} '.format(epoch,
                                                                                                           EPOCHS_FLOW,
                                                                                                           optim_flow.param_groups[0]['lr'],
                                                                                                           NB_IMGS_TRAIN_NI,
                                                                                                           NB_IMGS_TRAIN_NI,
                                                                                                           100.,
                                                                                                           (loss_epoch / len(data_train.dataloader_train_flow)).item()))

    fsvdd_loss.update_R(flow_wrapper.model, epoch, outputs)
    del outputs

    losses_flow.append(loss_epoch / len(data_train.dataloader_train_flow))

################################################################################

if not os.path.exists(results_path):
    os.makedirs(results_path)

torch.save({'epoch': epoch,
            'state_dict': img_map_wrapper.model.state_dict()},
           os.path.join(results_path, f'img_map_checkpoint_{EPOCHS_IMG_MAP}.pth'))

torch.save({'epoch': epoch,
            'state_dict': flow_wrapper.model.state_dict()},
           os.path.join(results_path, f'flow_checkpoint_{EPOCHS_FLOW}.pth'))

with open(os.path.join(results_path, 'vars.pkl'), 'wb') as f:
    pickle.dump({"train_ni_idx": data_train.train_ni_idx,
                 "train_cg_idx": data_train.train_cg_idx,
                 "normalize_img_map": data_train.normalize_img_map,
                 "normalize_flow": data_train.normalize_flow,
                 "C": fsvdd_loss.C,
                 "R": fsvdd_loss.R,
                 "W": fsvdd_loss.W}, f)

plt.figure(figsize=(14, 12))
ax1 = plt.subplot(121)
ax1.plot(range(EPOCHS_IMG_MAP), losses_img_map, label='Training loss img_map')
ax1.legend()

ax2 = plt.subplot(122)
ax2.plot(range(EPOCHS_FLOW), losses_flow, label='Training loss flow')
ax2.legend()

plt.savefig(os.path.join(results_path, 'loss_training.png'))
