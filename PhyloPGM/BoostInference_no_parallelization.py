import pandas as pd, numpy as np, time

global new_quant_test_df, quant_val_df
global pos_train_df, neg_train_df, curr_test_df
global pos_branch, neg_branch, pos_root_node, neg_root_node


class Booster:
    """Boosts Inference with PhyloPGM approach
    """

    def __init__(self,
                 df_val=None,
                 df_test=None,
                 root='hg38',
                 tree=None,
                 alpha=0.1,
                 tune=False,
                 missing_value=9
                 ):
        """

        :param df_val: pandas DataFrame [sp1, sp2, ..., spn, label], float
        :param df_test: pandas DataFrame [sp1, sp2, ..., spn], float
        :param root: root species name, str
        :param tree: child, parent pair s.t. tree[child]=parent, Dictionary
        :param alpha: branch likelihood coefficient, float
        :param tune: whether to tune alpha, bool
        :param missing_value: missing value indicator, float

        :return df_test: modified df_test with branch likelihood scores
        :return pred: combined prediction scores, array-like

        """
        self.df_val = df_val
        self.df_test = df_test
        self.root = root
        self.tree = tree
        self.alpha = alpha
        self.tune = tune
        self.missing_value = missing_value


    def boost(self):
        # assert df_val and df_test are float

        # round df_val
        self.df_val.iloc[:, :-1] = self.df_val.iloc[:, :-1].round(1)
        self.df_val = self.df_val.fillna(self.missing_value)

        # print('check num',
        #       self.df_val[(self.df_val['panTro4'] == 0.0)
        #                   & (self.df_val['HP'] == 0.1)
        #                   & (self.df_val['label'] == 1.)].shape[0]);
        #
        # print('check den',
        #       self.df_val[ (self.df_val['HP']==0.1)
        #       & (self.df_val['label']==1.)].shape[0]); exit(0)

        sp_list = self.df_val.columns[:-1]
        non_root_list = set(sp_list) - {self.root}
        # print('self.root:', self.root, 'sp_list:', sp_list,
        #       'non_root_list:', non_root_list); #exit(0)
        branch_list = ['branch_'+item+'_'+self.tree[item] for item in non_root_list]

        # round df_test
        self.df_test.iloc[:, :] = self.df_test.iloc[:, :].round(1)
        # print('df_test:', self.df_test.shape[0]);
        # print('df_test hg38:', self.df_test[self.df_test.hg38.notna()].shape[0]); exit(0)
        self.df_test = self.df_test.fillna(self.missing_value)

        pos_temp_df = self.df_val[self.df_val.label==1].copy()

        neg_temp_df = self.df_val[self.df_val.label==0].copy()

        global pos_branch, neg_branch, pos_root_node, neg_root_node

        pos_root_node = {}; neg_root_node = {}

        uniq_root_values = pos_temp_df.hg38.unique()
        pos_size = pos_temp_df.shape[0]
        for item in uniq_root_values:
            curr_prob = float(pos_temp_df[pos_temp_df['hg38']==item].shape[0])/float(pos_size)
            pos_root_node[str(item)] = curr_prob

        uniq_root_values = neg_temp_df.hg38.unique()
        neg_size = neg_temp_df.shape[0]
        for item in uniq_root_values:
            curr_prob = float(neg_temp_df[neg_temp_df['hg38']==item].shape[0])/float(neg_size)
            neg_root_node[str(item)] = curr_prob

        # print('pos_root_node:', pos_root_node)
        # print('neg_root_node:', neg_root_node)
        # exit(0)



        # create pos branch dict
        pos_branch = dict()
        # print('pos_temp_df:', pos_temp_df.shape)

        # create neg branch dict
        neg_branch = dict()
        # print('neg_temp_df:', neg_temp_df.shape)

        for br_item in branch_list:
            pos_branch[br_item] = {}
            neg_branch[br_item] = {}
            _, curr_br_child, curr_br_parent = br_item.split('_')

            # print('curr_br_child:', curr_br_child,
            #       'curr_br_parent:', curr_br_parent
            #       )

            # find unique parent values
            uniq_parent_values = pos_temp_df[curr_br_parent].unique()
            for item in uniq_parent_values:
                temp_temp = pos_temp_df[pos_temp_df[curr_br_parent]==item].copy()
                uniq_child_values = temp_temp[curr_br_child].unique()

                num_parent = temp_temp.shape[0]

                # compute probabilites
                for item_ in uniq_child_values:
                    num_child = temp_temp[temp_temp[curr_br_child]==item_].shape[0]
                    curr_prob = float(num_child)/float(num_parent)
                    # print('curr_prob:', curr_prob)
                    pos_branch[br_item][str(item_) + ' ' + str(item)] = curr_prob

            del temp_temp

            # find unique parent values
            uniq_parent_values = neg_temp_df[curr_br_parent].unique()
            for item in uniq_parent_values:
                temp_temp = neg_temp_df[neg_temp_df[curr_br_parent]==item].copy()
                uniq_child_values = temp_temp[curr_br_child].unique()

                num_parent = temp_temp.shape[0]

                # compute probabilites
                for item_ in uniq_child_values:
                    num_child = temp_temp[temp_temp[curr_br_child]==item_].shape[0]
                    curr_prob = float(num_child)/float(num_parent)
                    # print('curr_prob:', curr_prob)
                    neg_branch[br_item][str(item_) + ' ' + str(item)] = curr_prob

            # print('pos_branch:', pos_branch)
            # print('neg_branch:', neg_branch)
            # exit(0)







        if self.tune:
            # TODO select alpha by k-fold cross validation
            best_alpha = self.alpha
            pass
        else:
            best_alpha = self.alpha

        self.df_test = get_branch_scores(self.df_val, self.df_test, self.root, branch_list)

        print('ratio_root:', self.df_test[self.df_test.ratio_root.notna()].shape[0])
        print('sanity_sum:', self.df_test[self.df_test.sanity_sum.notna()].shape[0])

        self.df_test['pgm_pred'] = self.df_test['ratio_root']+(best_alpha*self.df_test['sanity_sum'])

        print('pgm_pred:', self.df_test[self.df_test.pgm_pred.notna()].shape[0])

        self.df_test['pgm_pred'] = self.df_test.pgm_pred.fillna(self.df_test.ratio_root)

        print('ratio_root pgm_pred:', self.df_test[self.df_test.pgm_pred.notna()].shape[0])

        self.df_test['pgm_pred'] = self.df_test.pgm_pred.fillna(self.df_test.sanity_sum)

        print('sanity_sum pgm_pred:', self.df_test[self.df_test.pgm_pred.notna()].shape[0])

        # exit(0)


        # pred = self.df_test['ratio_root']+(best_alpha*self.df_test['sanity_sum'])

        # return self.df_test, pred
        return self.df_test, self.df_test['pgm_pred'].values


def get_sum(row):
    s=0.
    for item in row.values:
        if not np.isnan(item):
            s+=item
    return s


def get_branch_scores(curr_train_df,
                      given_curr_test_df,
                      root,
                      branch_list
                      ):
    global pos_train_df, neg_train_df, curr_test_df, pos_root_node, neg_root_node

    curr_test_df = given_curr_test_df
    print('curr_test_df:', curr_test_df.shape)
    pos_train_df = curr_train_df[curr_train_df['label'] == 1]
    neg_train_df = curr_train_df[curr_train_df['label'] == 0]

    # root
    def process_root_branch(row):
        test_hg38 = row[root]
        curr_key = str(test_hg38)

        # calculate numerator
        if curr_key in pos_root_node.keys():
            num = pos_root_node[curr_key]
        else:
            return np.nan

        # calculate denominator
        if curr_key in neg_root_node.keys():
            den = neg_root_node[curr_key]
        else:
            return np.nan

        curr_root = np.log(num/den)
        return curr_root

    curr_test_df['ratio_root'] = curr_test_df.apply(process_root_branch,
                                                            axis=1
                                                            )
    print('Done process_root_branch')

    def process_non_root_branch(row, branch):
        child, parent = branch.split('_')[1:]
        test_child = row[child]
        test_parent = row[parent]



        curr_key = str(test_child) + ' ' + str(test_parent)

        # if child == 'panTro4' and parent=='HP' and test_child==0.0 and test_parent==0.1:
        #     print('test_child:', test_child,
        #           'test_parent:', test_parent
        #           )
        #     print('curr_key:', curr_key)
        #     print(pos_branch[branch][curr_key])
        #     print(neg_branch[branch][curr_key])
        #     exit(0)

        # calculate numerator
        if curr_key in pos_branch[branch].keys():
            num = pos_branch[branch][curr_key]
        else:
            return np.nan

        # calculate denominator
        if curr_key in neg_branch[branch].keys():
            den = neg_branch[branch][curr_key]
        else:
            return np.nan

        curr_second_part = np.log(num / den)
        return curr_second_part

    start = time.time()
    for item_id, item in enumerate(branch_list):
        if item_id % 50 == 0:
            print('item_id:', item_id,
                  'item:', item)
        curr_test_df[item] = curr_test_df.apply(process_non_root_branch,
                                                args=(item,),
                                                axis=1
                                                )
    stop = time.time()
    print('time taken:', stop-start)

    curr_test_df['sanity_sum'] = curr_test_df[branch_list].apply(get_sum,
                                                                         axis=1
                                                                         )

    return curr_test_df

