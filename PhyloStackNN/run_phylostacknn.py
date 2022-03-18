"""
Runs phylo ml stack
email: zaifyahsan@gmail.com
(c) Faizy Ahsan
"""

import pickle
import time, argparse
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV


# def get_recall_fpr(y, pred, req_fpr=0.01):
#     num_examples = len(y)
#     data = np.zeros(shape=(num_examples, 2))
#     data[:, 0] = pred
#     data[:, 1] = y
#
#     rev_sorted_data = data[np.argsort(-data[:, 0]), :]
#
#     for i in range(num_examples):
#         curr_pred = np.ones(shape=num_examples)
#         curr_pred[i + 1:] = 0.
#         tn, fp, fn, tp = confusion_matrix(rev_sorted_data[:, 1], curr_pred).ravel()
#         curr_tpr = tp / (tp + fn)
#         curr_fpr = fp / (fp + tn)
#
#         if curr_fpr > req_fpr:
#             break
#     return rev_sorted_data[i, 0], curr_tpr, curr_fpr


def do_meta_NN(train_X, train_y, test_X, test_y, if_save_model=False):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(train_X)
    imp_train_X = imp.transform(train_X)
    print('imp_train_X:', imp_train_X.shape)
    imp_test_X = imp.transform(test_X)
    print('imp_test_X:', imp_test_X.shape)

    meta_model = MLPClassifier()

    # do grid search
    nn_params = {'hidden_layer_sizes': [(32,), (100,), (64, 32)],
                 'alpha': [0.1, 1, 10],
                 'max_iter': [1500],
                 }
    meta_clf = GridSearchCV(meta_model, nn_params, cv=10, n_jobs=4,)
    meta_clf.fit(imp_train_X, train_y)

    meta_best_params = meta_clf.best_params_
    print('Meta Model NN best_params:', meta_best_params)

    meta_model.set_params(**meta_best_params)

    meta_model.fit(imp_train_X, train_y)
    pred_meta_y = meta_model.predict_proba(imp_test_X)[:, 1]

    pickle.dump(meta_model, open('meta_model_trained_on_validate_all_species_NN.pkl',
                                 'wb'))
    np.savetxt('pred_meta_model_trained_on_validate_all_species_NN_GS.txt',
               np.array([test_y, pred_meta_y]).T)

    test_auc_score = roc_auc_score(test_y, pred_meta_y)

    return test_auc_score, pred_meta_y


def scikitlearn_calc_auPRC(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


if __name__ == '__main__':
    start_time = time.time()

    np.random.seed(42)

    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    parser.add_argument("--train-data", type=str, default=None, help="Path to the train data file. Default: None")
    parser.add_argument("--validate-data", type=str, default=None,
                        help="Path to the validation data file. Default: None")
    parser.add_argument("--test-data", type=str, default=None, help="Path to the test data file. Default: None")
    parser.add_argument("--output-fname", type=str, default=None, help="Path to the output file. Default: None")

    # parser.add_argument("--info-tree", type=str, default=None,
    #                     help="Path to the file with tree, adjacency matrix and species indices. Default: None")
    # parser.add_argument('--rna', action='store_true', default=False,
    #                     help="whether to use with RNA-RBP data or ChIP-Seq data. Default: False")
    args = parser.parse_args()

    # load data
    df_validate = pd.read_csv(args.validate_data,
                                  index_col=0)
    df_test = pd.read_csv(args.test_data,
                          index_col=0)

    print('df_validate:', df_validate.shape,
          'df_test:', df_test.shape
          )
    # drop rows with no predictions on hg38 and no label values
    df_validate = df_validate[df_validate['hg38'].notna()]
    df_validate = df_validate[df_validate['y'].notna()]

    df_test = df_test[df_test['hg38'].notna()]
    df_test = df_test[df_test['y'].notna()]

    print('df_validate:', df_validate.shape,
          'df_test:', df_test.shape
          )
    #exit(0)


    pred_val_hg38 = df_validate['hg38'].values
    pred_test_hg38 = df_test['hg38'].values

    # create X,y
    v_X = df_validate.values[:, :-1]; v_y = df_validate.values[:, -1].astype(int)
    t_X = df_test.values[:, :-1]; t_y = df_test.values[:, -1].astype(int)
    print('v_X:', v_X.shape, 'v_y:', v_y.shape)
    print('t_X:', t_X.shape, 't_y:', t_y.shape)

    #print('t_y:', t_y)
    #print('pred_test_hg38:', pred_test_hg38); exit(0)

    base_auc = roc_auc_score(t_y, pred_test_hg38)
    base_aupr = scikitlearn_calc_auPRC(t_y,
                                       pred_test_hg38
                                       )

    y = t_y
    pred_hg38 = df_test['hg38'].values
    print('Run NN stack')
    nn_auc, pred_nn_stack = do_meta_NN(v_X, v_y, t_X, t_y)
    nn_aupr = scikitlearn_calc_auPRC(t_y,
                                     pred_nn_stack
                                     )
    print('Done. NN stack')

    
    df_stack = df_test.copy()

    df_stack['NN_stack'] = pred_nn_stack

    
    df_stack = df_stack[['hg38', 'NN_stack', 'y']]
    df_stack.to_csv(args.output_fname)


    print('Result val_size:', v_X.shape[0],
          'test_size:', t_X.shape[0],
          'Base Model Test AUC score:', base_auc,
          'NN AUC:', nn_auc,
          'Base Model AUPR:', base_aupr,
          'NN AUPR:', nn_aupr,
          )


    stop_time = time.time()

    print('Total Time Taken:', stop_time - start_time)





