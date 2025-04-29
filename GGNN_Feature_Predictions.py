import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from torch.utils.data import Dataset
from featurization import loadInputs_train,loadInputs_val,loadInputs_test,ToxicDataset,load_data
from sklearn.metrics import precision_recall_curve,mean_squared_error,r2_score,mean_absolute_error
from sklearn.metrics import roc_curve, auc, precision_score, recall_score
import xgboost as xgb
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from ggnn_model.ggnn_model import GGNN
import os
from data_deal import decrease_learning_rate
import pandas as pd
from torch.autograd import Variable
from tensorboardX import SummaryWriter
writer = SummaryWriter('loss')

def training(model, data,optimizer, criterion, args):
    model.train()
    total_loss = []
    for k in data:
        feature,A,y = k
        if len(y.shape) == 3:
            y = y.squeeze(1)
        model.zero_grad()
        padding = torch.zeros(len(feature), 50, args.hidden - 58)
        init_input = torch.cat((feature, padding), 2)
        init_input,A,feature,y = init_input.cuda(),A.cuda(),feature.cuda(),y.cuda()
        init_input,A,feature = Variable(init_input),Variable(A),Variable(feature)
        target = Variable(y)
        output,_ = model(init_input, feature, A)
        loss = criterion(output, target)
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        R2 = r2_score(y.cpu().detach().numpy(), output.cpu().detach().numpy())
    model.eval()
    val_total_loss = 0
    with torch.no_grad():
        for k in data:
            feature, A, y = k
            if len(y.shape) == 3:
                y = y.squeeze(1)
            padding = torch.zeros(len(feature), 50, args.hidden - 58)
            init_input = torch.cat((feature, padding), 2)
            init_input, A, feature, y = Variable(init_input.cuda()), Variable(A.cuda()), Variable(
                feature.cuda()), Variable(y.cuda())
            output, _ = model(init_input, feature, A)
            valid_loss = criterion(output, y)
            val_total_loss = val_total_loss+valid_loss
    return (sum(total_loss) / len(total_loss)),R2,val_total_loss

def evaluate(model, data, criterion, args):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for k in data:
            feature, A, y = k
            if len(y.shape) == 3:
                y = y.squeeze(1)
            padding = torch.zeros(len(feature), 50, args.hidden - 58)
            init_input = torch.cat((feature, padding), 2)
            init_input,A ,feature,y= Variable(init_input.cuda()),Variable(A.cuda()),Variable(feature.cuda()),Variable(y.cuda())
            output, feature = model(init_input, feature, A)
            total_loss.append((criterion(output,  y)).item())
            MSE = mean_squared_error(y.cpu().numpy(), output.cpu().numpy())
            RMSE = MSE ** 0.5
            R2 = r2_score(y.cpu().numpy(), output.cpu().numpy())
            mae = mean_absolute_error(y.cpu().numpy(), output.cpu().numpy())
    return (sum(total_loss) / len(total_loss)),RMSE,R2,feature,mae

def featurize(model,data, args):
    model.eval()
    i = 0
    with torch.no_grad():
        for k in data:
            feature, A, y = k
            padding = torch.zeros(len(feature), 50, args.hidden - 58)
            init_input = torch.cat((feature, padding), 2)
            init_input, A, feature, y = Variable(init_input.cuda()), Variable(A.cuda()), Variable(feature.cuda()), Variable(y.cuda())
            output, feature = model(init_input, feature, A)
            if i ==0:
                features = feature
            else:
                features = torch.cat((features,feature))
            i = i+1
    return features

def training_ggnn_class(model, data,optimizer, criterion, args):
    model.train()
    total_loss = []
    for k in data:
        feature,A,y = k
        model.zero_grad()
        if len(y.shape)==3:
            y = y.squeeze(1)
        padding = torch.zeros(len(feature), 50, args.hidden - 58)
        init_input = torch.cat((feature, padding), 2)
        init_input = init_input.cuda()
        init_input, A, feature, y = Variable(init_input.cuda()), Variable(A.cuda()), Variable(
            feature.cuda()), Variable(y.cuda())
        output, _ = model(init_input, feature, A)
        loss = criterion(output, y)
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    model.eval()
    val_total_loss = 0
    with torch.no_grad():
        for k in data:
            feature, A, y = k
            if len(y.shape) == 3:
                y = y.squeeze(1)
            padding = torch.zeros(len(feature), 50, args.hidden - 58)
            init_input = torch.cat((feature, padding), 2)
            init_input, A, feature, y = Variable(init_input.cuda()), Variable(A.cuda()), Variable(
                feature.cuda()), Variable(y.cuda())
            output, _ = model(init_input, feature, A)
            valid_loss = criterion(output, y)
            val_total_loss = val_total_loss+valid_loss
    return (sum(total_loss) / len(total_loss)),val_total_loss

def evaluate_classifier(model, data, criterion, args):
    model.eval()
    total_loss = []
    y_predict = []
    y_test = []
    with torch.no_grad():
        for k in data:
            feature, A, y = k
            if len(y.shape) == 3:
                y = y.squeeze(1)
            padding = torch.zeros(len(feature), 50, args.hidden - 58)
            init_input = torch.cat((feature, padding), 2)
            init_input, A, feature, y = Variable(init_input.cuda()), Variable(A.cuda()), Variable(
                feature.cuda()), Variable(y.cuda())
            output, _ = model(init_input, feature, A)
            total_loss.append((criterion(output,  y)).item())
            output = torch.sigmoid(output)
            output = output.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            for i in output:
                y_predict.append(i)
            for j in y:
                y_test.append(j)
    y_test = pd.DataFrame(y_test)
    y_predict = pd.DataFrame(y_predict)
    if y_test.shape[1]==1:
        fpr, tpr, threshold = roc_curve(y_test, y_predict)
        AUC = auc(fpr, tpr)
        output_tran = []
        for x in y_predict[0]:
            if x > 0.5:
                output_tran.append(1)
            else:
                output_tran.append(0)
        precision = precision_score(y_test, output_tran)
        recall = recall_score(y_test, output_tran)
    else:
        AUC_all = []
        precision_all = []
        recall_all = []
        for i in range(y_test.shape[1]):
            if max(y_test[i])==0:
                continue
            fpr, tpr, threshold = roc_curve(y_test[i], y_predict[i])
            AUC = auc(fpr, tpr)
            output_tran = []
            for x in y_predict[i]:
                if x > 0.5:
                    output_tran.append(1)
                else:
                    output_tran.append(0)
            precision = precision_score(y_test[i], output_tran)
            recall = recall_score(y_test[i], output_tran)
            AUC_all.append(AUC)
            precision_all.append(precision)
            recall_all.append(recall)
        AUC = np.mean(AUC_all)
        precision = np.mean(precision_all)
        recall = np.mean(recall_all)
    return (sum(total_loss) / len(total_loss)),AUC,precision,recall

def evaluate_test_metrics(model, data, criterion, args):
    model.eval()
    total_loss = []
    y_predict = []
    y_test = []
    with torch.no_grad():
        feature, A, y = data
        if len(y.shape)==3:
            y = y.squeeze(1)
        padding = torch.zeros(len(feature), 50, args.hidden - 58)
        init_input = torch.cat((feature, padding), 2)
        init_input, A, feature, y = Variable(init_input.cuda()), Variable(A.cuda()), Variable(
            feature.cuda()), Variable(y.cuda())
        output, _ = model(init_input, feature, A)
        total_loss.append((criterion(output,  y)).item())
        output = torch.sigmoid(output)
        output = output.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        for i in output:
            y_predict.append(i)
        for j in y:
            y_test.append(j)
    y_test = pd.DataFrame(y_test)
    y_predict = pd.DataFrame(y_predict)
    if y_test.shape[1]==1:
        fpr, tpr, threshold = roc_curve(y_test, y_predict)
        AUC = auc(fpr, tpr)
        output_tran = []
        for x in y_predict[0]:
            if x > 0.5:
                output_tran.append(1)
            else:
                output_tran.append(0)
        precision = precision_score(y_test, output_tran)
        recall = recall_score(y_test, output_tran)
    else:
        AUC_all = []
        precision_all = []
        recall_all = []
        for i in range(y_test.shape[1]):
            if max(y_test[i])==0 or max(y_predict[i])==0:
                continue
            fpr, tpr, threshold = roc_curve(y_test[i], y_predict[i])
            AUC = auc(fpr, tpr)
            output_tran = []
            for x in y_predict[i]:
                if x > 0.5:
                    output_tran.append(1)
                else:
                    output_tran.append(0)
            precision = precision_score(y_test[i], output_tran)
            recall = recall_score(y_test[i], output_tran)
            AUC_all.append(AUC)
            precision_all.append(precision)
            recall_all.append(recall)
        AUC = np.mean(AUC_all)
        precision = np.mean(precision_all)
        recall = np.mean(recall_all)
    return (sum(total_loss) / len(total_loss)),AUC,precision,recall

def xgb_regressor(X_train,y_train,X_val, y_val,X_test,y_test):
    from xgboost.sklearn import XGBRegressor
    if y_test.shape[-1]==1:
        model = XGBRegressor(
            learn_rate=0.1,
            max_depth=4,#4
            min_child_weight=10,
            gamma=1,#1
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.8,
            objective='reg:linear',
            n_estimators=2000,
            tree_method = 'gpu_hist',
            n_gpus = -1, eval_metric='rmse',
        )
        model.fit(X_train, y_train,eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_test)
        y_test = y_test.astype('float')
        MSE = mean_squared_error(y_test,y_pred)
        RMSE = MSE ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        return RMSE
    else:
        RMSEs = []
        if len(y_train.shape)==3:
            y_train = [x[0] for x in y_train]
            y_val = [x[0] for x in y_val]
            y_test = [x[0] for x in y_test]
            y_train = pd.DataFrame(y_train)
            y_val = pd.DataFrame(y_val)
            y_test = pd.DataFrame(y_test)
        for i in range(y_test.shape[1]):
            if float(max(y_val[i])) == 0 or float(max(y_train[i])) == 0 or float(max(y_test[i])) == 0:
                continue
            model = XGBRegressor(
                learn_rate=0.1,
                max_depth=4,  # 4
                min_child_weight=10,
                gamma=1,  # 1
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.8,
                objective='reg:linear',
                n_estimators=2000,
                tree_method='gpu_hist',
                n_gpus=-1
            )
            model.fit(X_train, [float(k) for k in y_train[i].astype(float).astype(int)], eval_set=[(X_val, [float(k) for k in y_val[i]])], eval_metric='rmse', verbose=False)
            y_pred = model.predict(X_test)
            y_test = y_test.astype('float')
            MSE = mean_squared_error(y_test[i], y_pred)
            RMSE = MSE ** 0.5
            mae = mean_absolute_error(y_test[i], y_pred)
            RMSEs.append(RMSE)
        return np.mean(RMSEs)

def regressors(X_train,y_train,X_val, y_val,X_test,y_test):
    scores = []
    if y_test.shape[-1]==1:
        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_test = y_test.astype('float')
        MSE = mean_squared_error(y_test,y_pred)
        RMSE = MSE ** 0.5
        type = 'GGNN+rf'
        scores.append([type,RMSE])
        clf = svm.SVR(C=0.8, cache_size=200, kernel='rbf', degree=3, epsilon=0.2)
        clf.fit(X_train, y_train)
        y_pre = clf.predict(X_test)
        MSE = mean_squared_error(y_test, y_pre)
        RMSE = MSE ** 0.5
        type = 'GGNN+svm'
        scores.append([type, RMSE])
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pre = knn.predict(X_test)
        MSE = mean_squared_error(y_test, y_pre)
        RMSE = MSE ** 0.5
        type = 'GGNN+knn'
        scores.append([type, RMSE])
        scores_df = pd.DataFrame(scores)
        return scores_df
    else:
        if len(y_train.shape)==3:
            y_train = [x[0] for x in y_train]
            y_val = [x[0] for x in y_val]
            y_test = [x[0] for x in y_test]
            y_train = pd.DataFrame(y_train)
            y_val = pd.DataFrame(y_val)
            y_test = pd.DataFrame(y_test)
        rf_rmse = []
        svm_rmse = []
        knn_rmse = []
        for i in range(y_test.shape[1]):
            if float(max(y_val[i])) == 0 or float(max(y_train[i])) == 0 or float(max(y_test[i])) == 0:
                continue
            rf = RandomForestRegressor()
            rf.fit(X_train, y_train[i])
            y_pred = rf.predict(X_test)
            y_test = y_test[i].astype('float')
            MSE = mean_squared_error(y_test[i], y_pred)
            RMSE = MSE ** 0.5
            rf_rmse.append(RMSE)
        type = 'GGNN+rf'
        scores.append([type, np.mean(rf_rmse)])
        for i in range(y_test.shape[1]):
            clf = svm.SVR(C=0.8, cache_size=200, kernel='rbf', degree=3, epsilon=0.2)
            clf.fit(X_train, y_train[i])
            y_pre = clf.predict(X_test)
            MSE = mean_squared_error(y_test[i], y_pre)
            RMSE = MSE ** 0.5
            type = 'GGNN+svm'
            scores.append([type, RMSE])
            knn = KNeighborsRegressor(n_neighbors=5)
            knn.fit(X_train, y_train[i])
            y_pre = knn.predict(X_test)
            MSE = mean_squared_error(y_test[i], y_pre)
            RMSE = MSE ** 0.5
            svm_rmse.append([type, RMSE])
        type = 'GGNN+svm'
        scores.append([type, np.mean(svm_rmse)])
        for i in range(y_test.shape[1]):
            knn = KNeighborsRegressor(n_neighbors=5)
            knn.fit(X_train, y_train[i])
            y_pre = knn.predict(X_test)
            MSE = mean_squared_error(y_test[i], y_pre)
            RMSE = MSE ** 0.5
            knn_rmse.append(RMSE)
        type = 'GGNN+knn'
        scores.append([type, np.mean(knn_rmse)])
        scores_df = pd.DataFrame(scores)
        return scores_df

def classifiers(X_train,y_train,X_val, y_val,X_test,y_test):
    if y_test.shape[-1]==1:
        scores = []
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        y_pre = rf.predict(X_test)
        fpr, tpr, threshold = roc_curve(y_test, y_pre)
        AUC = auc(fpr, tpr)
        type = 'ggnn+rf'
        scores.append([type, AUC])
        clf = svm.SVC(C=1, kernel='rbf', degree=3, gamma='auto',
                      coef0=0.0, shrinking=True, probability=False,
                      tol=1e-3, cache_size=200, class_weight=None,
                      verbose=False, max_iter=-1, decision_function_shape='ovr',
                      random_state=None)
        clf.fit(X_train, y_train)
        y_pre = clf.predict(X_test)
        fpr, tpr, threshold = roc_curve(y_test, y_pre)
        AUC = auc(fpr, tpr)
        type = 'ggnn+svm'
        scores.append([type, AUC])
        clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')
        clf.fit(X_train, y_train)
        y_pre = clf.predict(X_test)
        fpr, tpr, threshold = roc_curve(y_test, y_pre)
        AUC = auc(fpr, tpr)
        type = 'ggnn+rnn'
        scores.append([type, AUC])
        scores_df = pd.DataFrame(scores)
        return scores_df
    else:
        rf_auc = []
        svm_auc = []
        knn_auc = []
        scores = []
        if len(y_train.shape)==3:
            y_train = [x[0] for x in y_train]
            y_val = [x[0] for x in y_val]
            y_test = [x[0] for x in y_test]
            y_train = pd.DataFrame(y_train)
            y_val = pd.DataFrame(y_val)
            y_test = pd.DataFrame(y_test)
        for i in range(y_test.shape[1]):
            if float(max(y_train[i])) == 0 or float(max(y_test[i])) == 0 or float(min(y_test[i])) == 1:
                continue
            rf = RandomForestClassifier()
            rf.fit(X_train, y_train[i])
            y_pre = rf.predict(X_test)
            if float(max(y_pre)) == 0 or float(min(y_pre)) == 1:
                continue
            fpr, tpr, threshold = roc_curve([float(j) for j in y_test[i]], [float(k) for k in y_pre])
            AUC = auc(fpr, tpr)
            if AUC>0:
                rf_auc.append(AUC)
        type = 'ggnn+rf'
        scores.append([type, np.mean(rf_auc)])
        for i in range(y_test.shape[1]):
            clf = svm.SVC(C=1, kernel='rbf', degree=3, gamma='auto',
                          coef0=0.0, shrinking=True, probability=False,
                          tol=1e-3, cache_size=200, class_weight=None,
                          verbose=False, max_iter=-1, decision_function_shape='ovr',
                          random_state=None)
            clf.fit(X_train, y_train[i])
            y_pre = clf.predict(X_test)
            fpr, tpr, threshold = roc_curve([float(j) for j in y_test[i]], [float(k) for k in y_pre])
            AUC = auc(fpr, tpr)
            if AUC>0:
                svm_auc.append(AUC)
        type = 'ggnn+svm'
        scores.append([type, np.mean(svm_auc)])
        for i in range(y_test.shape[1]):
            clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')
            clf.fit(X_train, y_train[i])
            y_pre = clf.predict(X_test)
            fpr, tpr, threshold = roc_curve([float(j) for j in y_test[i]], [float(k) for k in y_pre])
            AUC = auc(fpr, tpr)
            if AUC>0:
                knn_auc.append(AUC)
        type = 'ggnn+knn'
        scores.append([type, np.mean(knn_auc)])
        scores_df = pd.DataFrame(scores)
        return scores_df

def xgb_classifier(X_train,y_train,X_val, y_val,X_test,y_test):
    if y_test.shape[-1]==1:
        xgb_gbc = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
               colsample_bytree=1, gamma=1, learning_rate=0.1, max_delta_step=0,
               max_depth=4, min_child_weight=8, n_estimators=2000,
               n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
               silent=True, subsample=0.8,tree_method='gpu_hist',n_gpus=-1,eval_metric = 'auc',early_stopping_rounds=300)
        xgb_gbc.fit(X_train,y_train,eval_set = [(X_val,y_val)], verbose=False)
        pre_pro = xgb_gbc.predict_proba(X_test)[:,1]
        fpr,tpr,threshold = roc_curve([float(i) for i in y_test],pre_pro)
        AUC = auc(fpr,tpr)
        return AUC
    else:
        aucs = []
        if len(y_train.shape)==3:
            y_train = [x[0] for x in y_train]
            y_val = [x[0] for x in y_val]
            y_test = [x[0] for x in y_test]
            y_train = pd.DataFrame(y_train)
            y_val = pd.DataFrame(y_val)
            y_test = pd.DataFrame(y_test)
        for i in range(y_test.shape[1]):
            if float(max(y_val[i])) == 0 or float(max(y_train[i])) == 0 or float(max(y_test[i])) == 0:
                continue
            xgb_gbc = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                        colsample_bytree=1, gamma=1, learning_rate=0.1, max_delta_step=0,
                                        max_depth=4, min_child_weight=8, n_estimators=2000,
                                        n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
                                        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                                        silent=True, subsample=0.8, tree_method='gpu_hist', n_gpus=-1, eval_metric='auc')
            xgb_gbc.fit(X_train, y_train[i].astype(float).astype(int), eval_set=[(X_val, y_val[i].astype(float).astype(int))], verbose=False)
            pre_pro = xgb_gbc.predict_proba(X_test)[:, 1]
            fpr, tpr, threshold = roc_curve([float(j) for j in y_test[i]], pre_pro)
            AUC = auc(fpr, tpr)
            aucs.append(AUC)
        return np.mean(aucs)

def get_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--folder', type=str, default="Data/BACE")
    parser.add_argument('--hidden', type=int, default=512,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_size', type=float, default=128)
    parser.add_argument('--natom', type=float, default=50)
    parser.add_argument('--nclass', type=float, default=1)
    parser.add_argument('--benchmark',type=str,default='BACE')
    parser.add_argument('--type', type=str, default="classification")
    args = parser.parse_args()
    return args

def scores(args, device):
    results = []
    dataset_path = args.folder
    args.dataset = args.folder 
    load_data(args)
    feature_train, a_train, y_train = loadInputs_train(args)
    feature_val, a_val, y_val = loadInputs_val(args)
    feature_test, a_test, y_test = loadInputs_test(args)
    args.nclass = y_test.shape[-1]
    model = GGNN(args.natom, args.hidden, args.nclass, args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_dataset = ToxicDataset(feature_train, a_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = ToxicDataset(feature_val, a_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataset = ToxicDataset(feature_test, a_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    if args.type == 'regression':
        args.metric = 'RMSE'
        criterion = nn.MSELoss()
        for epoch in range(args.epochs):
            train_loss, train_R2, val_total_loss = training(model, train_loader, optimizer, criterion, args)
            val_loss, val_RMSE, val_R2, _, _ = evaluate(model, val_loader, criterion, args)
            if epoch % 4 == 0 and epoch != 0:
                decrease_learning_rate(optimizer, decrease_by=0.001)
        test_loss, test_RMSE, test_R2, _, MAE = evaluate(model, test_loader, criterion, args)
        # GGNN Score
        results.append(["GGNN", test_RMSE])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        xgb_train_feature = featurize(model, train_loader, args)
        xgb_val_feature = featurize(model, val_loader, args)
        xgb_test_feature = featurize(model, test_loader, args)
        xgb_train_feature = xgb_train_feature.cpu().numpy()
        xgb_val_feature = xgb_val_feature.cpu().numpy()
        xgb_test_feature = xgb_test_feature.cpu().numpy()
        # Load descriptor and fragment features
        desc_train = np.load(os.path.join(dataset_path, 'train_desc.npy'))
        frag_train = np.load(os.path.join(dataset_path, 'train_frag.npy'))
        desc_val = np.load(os.path.join(dataset_path, 'val_desc.npy'))
        frag_val = np.load(os.path.join(dataset_path, 'val_frag.npy'))
        desc_test = np.load(os.path.join(dataset_path, 'test_desc.npy'))
        frag_test = np.load(os.path.join(dataset_path, 'test_frag.npy'))
        # Handle mismatches
        min_train = min(xgb_train_feature.shape[0], desc_train.shape[0], frag_train.shape[0])
        min_val = min(xgb_val_feature.shape[0], desc_val.shape[0], frag_val.shape[0])
        min_test = min(xgb_test_feature.shape[0], desc_test.shape[0], frag_test.shape[0])
        xgb_train_feature = xgb_train_feature[:min_train]
        desc_train = desc_train[:min_train]
        frag_train = frag_train[:min_train]
        xgb_val_feature = xgb_val_feature[:min_val]
        desc_val = desc_val[:min_val]
        frag_val = frag_val[:min_val] 
        xgb_test_feature = xgb_test_feature[:min_test]
        desc_test = desc_test[:min_test]
        frag_test = frag_test[:min_test]
        # Concatenate features
        xgb_train_feature = np.concatenate([xgb_train_feature, desc_train, frag_train], axis=1)
        xgb_val_feature = np.concatenate([xgb_val_feature, desc_val, frag_val], axis=1)
        xgb_test_feature = np.concatenate([xgb_test_feature, desc_test, frag_test], axis=1)
        # XGB
        xgb_rmse = xgb_regressor(xgb_train_feature, y_train, xgb_val_feature, y_val, xgb_test_feature, y_test)
        results.append(["GGNN+XGB", xgb_rmse])
        # RF, SVM, KNN
        other_models_df = regressors(xgb_train_feature, y_train, xgb_val_feature, y_val, xgb_test_feature, y_test)
        for idx, row in other_models_df.iterrows():
            model_name = row[0]
            score = row[1]
            results.append([model_name, score])
    else:
        args.metric = 'AUC'
        criterion = nn.BCEWithLogitsLoss()
        for epoch in range(args.epochs):
            train_loss, val_total_loss = training_ggnn_class(model, train_loader, optimizer, criterion, args)
            val_loss, val_AUC, val_precision, val_recall = evaluate_classifier(model, val_loader, criterion, args)
            if epoch % 4 == 0 and epoch != 0:
                decrease_learning_rate(optimizer, decrease_by=0.001)
        test_loader_xgb = (torch.from_numpy(np.float32(feature_test)), torch.from_numpy(np.float32(a_test)), torch.from_numpy(np.float32(y_test)))
        test_loss, test_AUC, test_precision, test_recall = evaluate_test_metrics(model, test_loader_xgb, criterion, args)
        results.append(["GGNN", test_AUC])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        xgb_train_feature = featurize(model, train_loader, args)
        xgb_val_feature = featurize(model, val_loader, args)
        xgb_test_feature = featurize(model, test_loader, args)
        xgb_train_feature = xgb_train_feature.cpu().numpy()
        xgb_val_feature = xgb_val_feature.cpu().numpy()
        xgb_test_feature = xgb_test_feature.cpu().numpy()
        desc_train = np.load(os.path.join(dataset_path, 'train_desc.npy'))
        frag_train = np.load(os.path.join(dataset_path, 'train_frag.npy'))
        desc_val = np.load(os.path.join(dataset_path, 'val_desc.npy'))
        frag_val = np.load(os.path.join(dataset_path, 'val_frag.npy'))
        desc_test = np.load(os.path.join(dataset_path, 'test_desc.npy'))
        frag_test = np.load(os.path.join(dataset_path, 'test_frag.npy'))
        min_train = min(xgb_train_feature.shape[0], desc_train.shape[0], frag_train.shape[0])
        min_val = min(xgb_val_feature.shape[0], desc_val.shape[0], frag_val.shape[0])
        min_test = min(xgb_test_feature.shape[0], desc_test.shape[0], frag_test.shape[0])
        xgb_train_feature = np.concatenate([xgb_train_feature[:min_train], desc_train[:min_train], frag_train[:min_train]], axis=1)
        xgb_val_feature = np.concatenate([xgb_val_feature[:min_val], desc_val[:min_val], frag_val[:min_val]], axis=1)
        xgb_test_feature = np.concatenate([xgb_test_feature[:min_test], desc_test[:min_test], frag_test[:min_test]], axis=1)
        xgb_auc = xgb_classifier(xgb_train_feature, y_train, xgb_val_feature, y_val, xgb_test_feature, y_test)
        results.append(["GGNN+XGB", xgb_auc])
        other_models_df = classifiers(xgb_train_feature, y_train, xgb_val_feature, y_val, xgb_test_feature, y_test)
        for idx, row in other_models_df.iterrows():
            model_name = row[0]
            score = row[1]
            results.append([model_name, score])
    final_df = pd.DataFrame(results, columns=["Model", "Score"])
    return final_df

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_argv()
    df = scores(args, device)
    output_file = args.benchmark + '_ggnn_scores.csv'
    df.to_csv(output_file, index=False)
    print(f"Output Metrics saved to {output_file}")