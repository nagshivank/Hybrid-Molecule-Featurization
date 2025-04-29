import numpy as np
import torch
import argparse
import os
import pandas as pd
from sklearn.metrics import roc_curve, auc, mean_squared_error
from tensorboardX import SummaryWriter
from torch import optim, nn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from data_deal import decrease_learning_rate
from gcn_model.gcn_model import GCN
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings("ignore")
from gcn_model.gcn_xgboost_scores import get_feature
from gcn_model.gcn_training import training,evaluate,training_classing,evaluate_classion,evaluate_test_scros
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from featurization import loadInputs_train,loadInputs_val,loadInputs_test,ToxicDataset,load_data
writer = SummaryWriter('loss')

# Regressor function
def regression(X_train, y_train, X_val, y_val, X_test, y_test):
    scores = []
    if y_test.shape[-1] == 1:
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        # GCN+rf
        rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        scores.append(["GCN+rf", rmse])
        # GCN+svm
        svm_model = make_pipeline(StandardScaler(), svm.SVR(C=10, epsilon=0.1, kernel='rbf'))
        svm_model.fit(X_train, y_train)
        y_pred = svm_model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        scores.append(["GCN+svm", rmse])
        # GCN+knn
        knn_model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7))
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        scores.append(["GCN+knn", rmse])
        # GCN+xgboost
        xgb_model = xgb.XGBRegressor(
            learning_rate=0.1,
            max_depth=4,
            min_child_weight=10,
            gamma=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.8,
            n_estimators=2000,
            tree_method='gpu_hist',
            n_gpus=-1,
            eval_metric='rmse',
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = xgb_model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        scores.append(["GCN+xgboost", rmse])
        return pd.DataFrame(scores)
    else:
        raise NotImplementedError("Multi-task regression not supported by this XGBoost Model")

# Classifier function
def classification(X_train, y_train, X_val, y_val, X_test, y_test):
    scores = []
    if y_test.shape[-1] == 1:
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        # RF
        rf = RandomForestClassifier(n_estimators=300, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test.astype(int), y_pred.astype(int))
        auc_rf = auc(fpr, tpr)
        scores.append(["GCN+rf", auc_rf])
        # SVM
        svm_model = make_pipeline(StandardScaler(), svm.SVC(C=1, kernel='rbf', probability=True))
        svm_model.fit(X_train, y_train)
        y_pred = svm_model.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test.astype(int), y_pred.astype(int))
        auc_svm = auc(fpr, tpr)
        scores.append(["GCN+svm", auc_svm])
        # KNN
        knn_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test.astype(int), y_pred.astype(int))
        auc_knn = auc(fpr, tpr)
        scores.append(["GCN+knn", auc_knn])
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            learning_rate=0.1,
            max_depth=4,
            min_child_weight=8,
            gamma=1,
            subsample=0.8,
            n_estimators=2000,
            tree_method='gpu_hist',
            n_gpus=-1,
            eval_metric='auc',
            use_label_encoder=False,
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = xgb_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test.astype(int), y_pred)
        auc_xgb = auc(fpr, tpr)
        scores.append(["GCN+xgboost", auc_xgb])
        return pd.DataFrame(scores)
    else:
        raise NotImplementedError("Multi-task regression not supported by this XGBoost Model")

def get_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--folder', type=str, default="Data/BACE")
    parser.add_argument('--hidden', type=int, default=512, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--batch_size', type=float, default=128)
    parser.add_argument('--natom', type=float, default=58)
    parser.add_argument('--benchmark',type=str,default='bace')
    parser.add_argument('--nclass', type=float, default=1)
    parser.add_argument('--type', type=str, default="classification")
    args = parser.parse_args()
    return args

# Predictor evaluation function
def scores(gcn_scores, xgb_scores, names, args, device):
    args.dataset = args.folder
    load_data(args)
    feature_train, a_train, y_train = loadInputs_train(args)
    feature_val, a_val, y_val = loadInputs_val(args)
    feature_test, a_test, y_test = loadInputs_test(args)
    args.nclass = y_test.shape[-1]
    model = GCN(args.natom, args.hidden, args.nclass, args.dropout).to(device)
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
            train_loss, train_R2, val_total_loss = training(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_RMSE, val_R2, _, _ = evaluate(model, val_loader, criterion, device)
            if epoch % 4 == 0 and epoch != 0:
                decrease_learning_rate(optimizer, decrease_by=0.001)
        test_loss, test_RMSE, test_R2, _, MAE = evaluate(
            model, test_loader, criterion, device
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        xgb_train_feature = get_feature(model, train_loader, device).cpu().numpy()
        xgb_val_feature   = get_feature(model, val_loader, device).cpu().numpy()
        xgb_test_feature  = get_feature(model, test_loader, device).cpu().numpy()
        desc_train = np.load(args.dataset + '/train_desc.npy')
        frag_train = np.load(args.dataset + '/train_frag.npy')
        desc_val   = np.load(args.dataset + '/val_desc.npy')
        frag_val   = np.load(args.dataset + '/val_frag.npy')
        desc_test  = np.load(args.dataset + '/test_desc.npy')
        frag_test  = np.load(args.dataset + '/test_frag.npy')
        min_train = min(xgb_train_feature.shape[0], desc_train.shape[0], frag_train.shape[0])
        min_val = min(xgb_val_feature.shape[0], desc_val.shape[0], frag_val.shape[0])
        min_test = min(xgb_test_feature.shape[0], desc_test.shape[0], frag_test.shape[0])
        xgb_train_feature = np.concatenate([xgb_train_feature[:min_train], desc_train[:min_train], frag_train[:min_train]], axis=1)
        xgb_val_feature   = np.concatenate([xgb_val_feature[:min_val],   desc_val[:min_val],   frag_val[:min_val]], axis=1)
        xgb_test_feature  = np.concatenate([xgb_test_feature[:min_test], desc_test[:min_test], frag_test[:min_test]], axis=1)
        rmse_df = regression(xgb_train_feature, y_train, xgb_val_feature, y_val, xgb_test_feature, y_test)
        return rmse_df
    else:
        args.metric = 'AUC'
        criterion = nn.BCEWithLogitsLoss()
        for epoch in range(args.epochs):
            train_loss, val_total_loss = training_classing(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_AUC, val_precision, val_recall = evaluate_classion(
                model, val_loader, criterion, device
            )
            if epoch % 4 == 0 and epoch != 0:
                decrease_learning_rate(optimizer, decrease_by=0.001)
        test_loader_xgb = (
            torch.from_numpy(np.float32(feature_test)),
            torch.from_numpy(np.float32(a_test)),
            torch.from_numpy(np.float32(y_test))
        )
        test_loss, test_AUC, test_precision, test_recall = evaluate_test_scros(
            model, test_loader_xgb, criterion, device
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        xgb_train_feature = get_feature(model, train_loader, device).cpu().numpy()
        xgb_val_feature   = get_feature(model, val_loader, device).cpu().numpy()
        xgb_test_feature  = get_feature(model, test_loader, device).cpu().numpy()
        desc_train = np.load(args.dataset + '/train_desc.npy')
        frag_train = np.load(args.dataset + '/train_frag.npy')
        desc_val   = np.load(args.dataset + '/val_desc.npy')
        frag_val   = np.load(args.dataset + '/val_frag.npy')
        desc_test  = np.load(args.dataset + '/test_desc.npy')
        frag_test  = np.load(args.dataset + '/test_frag.npy')
        min_train = min(xgb_train_feature.shape[0], desc_train.shape[0], frag_train.shape[0])
        min_val = min(xgb_val_feature.shape[0], desc_val.shape[0], frag_val.shape[0])
        min_test = min(xgb_test_feature.shape[0], desc_test.shape[0], frag_test.shape[0])
        xgb_train_feature = np.concatenate([xgb_train_feature[:min_train], desc_train[:min_train], frag_train[:min_train]], axis=1)
        xgb_val_feature   = np.concatenate([xgb_val_feature[:min_val],   desc_val[:min_val],   frag_val[:min_val]], axis=1)
        xgb_test_feature  = np.concatenate([xgb_test_feature[:min_test], desc_test[:min_test], frag_test[:min_test]], axis=1)
        auc_df = classification(xgb_train_feature, y_train, xgb_val_feature, y_val, xgb_test_feature, y_test)
        return auc_df

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_argv()
    gcn_scores = []
    xgb_scores = []
    names = locals()
    df = scores(gcn_scores, xgb_scores, names, args, device)
    output_file = args.benchmark + "_gcn_scores.csv"
    df.to_csv(output_file, index=False)
    print(f"Output Metrics saved to {output_file}")
