import pandas
import os
import torch as tr
import torch.nn as nn
import torch.utils.data as tr_data
import sklearn.metrics as metrics
from tqdm import tqdm
from MAT import DEVICE, logger


def mat_train(model: nn.Module,
              train_set: tr_data.Dataset,
              valid_set: tr_data.Dataset = None,
              weights_save_name: str = 'weight',
              only_save_best_of_best: bool = True,
              save_before_early_stop: bool = False,
              curve_save_name: str = None,
              learning_rate: float = 1e-3,
              weight_decay: float = 0.,
              batch_size: int = 32,
              max_epoch: int = 100,
              class_weight: tr.Tensor = None,
              patience: int = 10
              ) -> None:
    # init train and valid data loader objects
    train_loader = tr_data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    if valid_set is not None:
        valid_loader = tr_data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # init loss function object
    loss_function = nn.CrossEntropyLoss(reduction='none') if class_weight is None \
        else nn.CrossEntropyLoss(weight=class_weight, reduction='none')

    # init optimizer object
    optimizer = tr.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # set device
    model = model.to(DEVICE)

    best_metric = 0.
    org_patience = patience
    last_epoch_save_name = None

    if curve_save_name is not None:
        curve = []

    for epoch in tqdm(range(1, max_epoch + 1)):
        model = model.train()

        epoch_loss = 0.
        n_batch = 0
        ypred_train = []
        ytrue_train = []

        for batch_idx, (data, label) in enumerate(train_loader):
            """
            Format of data and label are as follows: 
                data: array shape [num windows, length, channel] or list/tuple of arrays
                label: array shape [num windows, 3]
                    in which 3 columns are:
                        - categorical label
                        - sample weight
                        - multi-task mask (0 or 1, samples having mask=0 will be in the first tensor of model's output, 
                            and mask=1 will be in the second one)
            """
            n_batch += 1

            # extract components in label tensor:
            # sample weight
            sample_weight = label[:, 1].float()
            sample_weight = sample_weight / sample_weight.mean()

            # multi-task mask
            multitask_mask = label[:, 2].bool()

            # label
            label = label[:, 0].long()

            optimizer.zero_grad()
            # source output, target output
            train_output_0, train_output_1 = model(data, multitask_mask=multitask_mask)

            loss0 = loss_function(train_output_0, label[~multitask_mask]) * sample_weight[~multitask_mask]
            loss1 = loss_function(train_output_1, label[multitask_mask]) * sample_weight[multitask_mask]
            loss = loss0.mean() + loss1.mean()

            loss.backward()
            optimizer.step()

            # record train loss and train ytrue, ypred
            with tr.no_grad():
                epoch_loss += loss
                ypred_train.append(train_output_1)
                ytrue_train.append(label[multitask_mask])

        # epoch evaluation
        with tr.no_grad():
            # calculate scores on training set
            epoch_loss /= n_batch
            ytrue_train = tr.cat(ytrue_train, 0).cpu().view(-1).numpy()
            ypred_train = tr.cat(ypred_train, 0).cpu().argmax(1).view(-1).numpy()

            train_f1 = metrics.f1_score(ytrue_train, ypred_train, average='macro')
            train_acc = metrics.accuracy_score(ytrue_train, ypred_train)

            # evaluate on valid set
            if valid_set is not None:
                model.eval()

                val_epoch_loss = 0.
                ypred_valid = []
                ytrue_valid = []
                n_batch = 0

                for (val_data, val_label) in valid_loader:
                    n_batch += 1
                    val_label = val_label.long()

                    _, val_output = model(val_data, tr.ones(len(val_data)).bool())
                    val_loss = loss_function(val_output, val_label).mean()

                    val_epoch_loss += val_loss
                    ypred_valid.append(val_output)
                    ytrue_valid.append(val_label)

                val_epoch_loss /= n_batch
                ytrue_valid = tr.cat(ytrue_valid, 0).cpu().view(-1).numpy()
                ypred_valid = tr.cat(ypred_valid, 0).cpu()
                ypred_valid = ypred_valid[:, :6]
                ypred_valid = ypred_valid.argmax(1).view(-1).numpy()

                val_f1 = metrics.f1_score(ytrue_valid, ypred_valid, average='macro')
                val_acc = metrics.accuracy_score(ytrue_valid, ypred_valid)
            else:
                val_epoch_loss = epoch_loss
                val_f1 = train_f1
                val_acc = train_acc

            logger.info("train loss: %.4f\ttrain f1: %.4f\t train acc: %.4f\n"
                        "  val loss: %.4f\t  val f1: %.4f\t   val acc: %.4f"
                        % (epoch_loss, train_f1, train_acc, val_epoch_loss, val_f1, val_acc))

            if curve_save_name is not None:
                curve.append({
                    "train_loss": epoch_loss.item(),
                    "valid_loss": val_epoch_loss.item(),
                    "train_f1": train_f1,
                    "valid_f1": val_f1,
                    "train_acc": train_acc,
                    "valid_acc": val_acc
                })

            # model checkpoint
            if val_f1 > best_metric:
                epoch_name = "%s_%d_%.6f" % (weights_save_name, epoch, val_f1)
                patience = org_patience
                logger.info(f"f1 improved from {best_metric} to {val_f1}, save to {epoch_name}")
                tr.save(model.state_dict(), epoch_name)
                best_metric = val_f1

                if last_epoch_save_name is not None and os.path.exists(last_epoch_save_name) and only_save_best_of_best:
                    os.remove(last_epoch_save_name)
                last_epoch_save_name = epoch_name

            # early stopping
            else:
                patience -= 1
                if patience <= 0:
                    if save_before_early_stop:
                        epoch_name = "%s_%d_%.6f" % (weights_save_name, epoch, val_f1)
                        tr.save(model.state_dict(), epoch_name + '-early_stop')
                    logger.info('STOPPED')
                    break

    if curve_save_name is not None:
        training_curve = pandas.DataFrame.from_records(curve)
        training_curve.to_csv(f"{curve_save_name}.csv", index=False)
