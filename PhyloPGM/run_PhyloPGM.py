from BoostInference_no_parallelization import Booster
import sys, pandas as pd, numpy as np
import glob, pickle
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


if len(sys.argv)<5:
    print('python file.py df-val-PhyloPGM-input df-test-PhyloPGM-output info_tree fname_df_pgm_output')
    exit(0)

fname_dtrain = sys.argv[1]
fname_dtest = sys.argv[2]
info_tree = sys.argv[3]
fname_df_pgm_output = sys.argv[4]
# given_pseudo_count = float(sys.argv[3])


print('fname_dtrain:', fname_dtrain,
      'fname_dtest:', fname_dtest,
      # 'given_pseudo_count:', given_pseudo_count
      )

dtrain = pd.read_csv(fname_dtrain, index_col=0)
list_species = list(dtrain.columns[:-1])
print('list_species:', len(list_species))
dtrain.columns = list_species + ['label']

# print(dtrain.head()); exit(0)

dtest = pd.read_csv(fname_dtest, index_col=0)
dtest.columns = list_species + ['label']

print('dtrain:', dtrain.shape)
print('dtest:', dtest.shape)

dtest = dtest[dtest.hg38.notna()]
print('dtest not hg38:', dtest.shape)

# exit(0)

given_tree = pickle.load(open(info_tree, 'rb'))

tree = {}
# for k, v in given_tree['tree'].items():
for k, v in given_tree.items():
    # tree[k] = v[0]
    tree[k] = v

print('tree:', len(tree))
# exit(0)

alpha = 0.1


num_pos_train = dtrain[dtrain.label==1].shape[0]
num_neg_train = dtrain[dtrain.label==0].shape[0]
num_pos_test = dtest[dtest.label==1].shape[0]
num_neg_test = dtest[dtest.label==0].shape[0]

print('num_pos_train:', num_pos_train,
      'num_neg_train:', num_neg_train,
      'num_pos_test:', num_pos_test,
      'num_neg_test:', num_neg_test
      )

base_auc = roc_auc_score(dtest.label, dtest.hg38)


def scikitlearn_calc_auPRC(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


base_aupr = scikitlearn_calc_auPRC(dtest.label, dtest.hg38)

print('base_auc:', base_auc,
      'base_aupr:', base_aupr
      )
# exit(0)

def get_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


BATCH_SIZE = 100000
tmp_mod_df_test_final = pd.DataFrame()

for counter, batch_ids in enumerate(get_batch(range(dtest.shape[0]), BATCH_SIZE)):

    curr_df = dtest.iloc[batch_ids, :].copy()
    print('curr_df:', curr_df.shape)

    model = Booster(df_val=dtrain.copy(),
                        df_test=curr_df.iloc[:, :-1].copy(),  # do not send test labels
                        tree=tree,
                        alpha=alpha,
                        # pseudo_count=given_pseudo_count
                        )

    mod_df_test, pred = model.boost()

    mod_df_test['original_hg38'] = curr_df['hg38']
    mod_df_test['label'] = curr_df['label']

    # tmp_mod_df_test_final = pd.concat([tmp_mod_df_test_final, curr_df[['hg38', 'label']]])
    tmp_mod_df_test_final = pd.concat([tmp_mod_df_test_final, mod_df_test])

print('tmp_mod_df_test_final:', tmp_mod_df_test_final.shape)
print('tmp_mod_df_test_final notna:', tmp_mod_df_test_final[tmp_mod_df_test_final.pgm_pred.notna()].shape)

# print(tmp_mod_df_test_final[tmp_mod_df_test_final.pgm_pred.isna()][['hg38', 'ratio_root', 'pgm_pred']]);
# print(tmp_mod_df_test_final.hg38.unique())
# exit(0)
# tmp_mod_df_test_final = tmp_mod_df_test_final[tmp_mod_df_test_final.pgm_pred.notna()]
# print('tmp_mod_df_test_final:', tmp_mod_df_test_final.shape)
# print(tmp_mod_df_test_final[~tmp_mod_df_test_final.pgm_pred.notna()])
# tmp_mod_df_test_final.to_csv('check.csv')
# exit(0)

tmp_mod_df_test_final = tmp_mod_df_test_final[tmp_mod_df_test_final.pgm_pred.notna()]
print('tmp_mod_df_test_final:', tmp_mod_df_test_final.shape)

base_auc = roc_auc_score(tmp_mod_df_test_final.label,
                         tmp_mod_df_test_final.original_hg38
                         )
pgm_auc = roc_auc_score(tmp_mod_df_test_final.label,
                        tmp_mod_df_test_final.pgm_pred)

base_aupr = scikitlearn_calc_auPRC(tmp_mod_df_test_final.label,
                         tmp_mod_df_test_final.original_hg38
                         )

pgm_aupr = scikitlearn_calc_auPRC(tmp_mod_df_test_final.label,
                        tmp_mod_df_test_final.pgm_pred)

print('Refined base_auc:', base_auc, 'pgm_auc:', pgm_auc,
      'base_aupr:', base_aupr, 'pgm_aupr:', pgm_aupr
      )
#tmp_mod_df_test_final.to_csv(fname+'/df-pgm-refined.csv')
tmp_mod_df_test_final.to_csv(fname_df_pgm_output)
