import transtab
import torch

if __name__ == '__main__':
    # load multiple datasets by passing a list of data names
    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
            = transtab.load_data(['./data/dataset1','./data/dataset2' ,'./data/dataset3']) # ,'.\\data\\dataset2'过大，跑不动
    # 预测目标需要自己指定列名为target_label

    # build transtab classifier model
    # model = transtab.build_classifier(cat_cols, num_cols, bin_cols)
    model, collate_fn = transtab.build_contrastive_learner(
        cat_cols, num_cols, bin_cols,
        supervised=False,
        num_partition=4,
        overlap_ratio=0.5
    )
    # model = torch.nn.DataParallel(model)  # 并不兼容多GPU并行训练

    # specify training arguments, take validation loss for early stopping
    training_arguments = {
        'num_epoch': 1000,
        'batch_size':64,
        'lr':1e-4,
        'eval_metric': 'val_loss',
        'eval_less_is_better': True,
        'output_dir': './checkpoint'
    }

    transtab.train(model, trainset, valset[0],collate_fn=collate_fn, **training_arguments)

    # load the pretrained model and finetune on a target dataset
    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
        = transtab.load_data('credit-approval')

    # build transtab classifier model, and load from the pretrained dir
    model = transtab.build_classifier(checkpoint='./checkpoint')

    # update model's categorical/numerical/binary column dict
    model.update({'cat': cat_cols, 'num': num_cols, 'bin': bin_cols})

    # 监督学习
    # ypred = transtab.predict(model, x_test) # 注意：预测分类标签返回值在[0,1]之间，需要手动修改
    # print(transtab.evaluate(ypred, y_test, seed=123, metric='auc')) # 评估
