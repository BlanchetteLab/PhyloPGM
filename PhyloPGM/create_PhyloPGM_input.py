# converts species predictions into dataframe
# Input: region,species,label, prediction_score
#        chr1:345-347, hg38, 0, 0.45
#        .....
# Output: _, species1, species2, ..., speciesn, y
#         chr1:345-347, 0.23, 0.42, ..., 0.21, 0
# email: zaifyahsan@gmail.com
# (c) Faizy Ahsan

import pandas as pd, numpy as np, pickle as pkl
import sys
import swifter
global df_req
from multiprocessing import Pool




def get_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


if len(sys.argv) < 5:
    print('python file.py pred_data output_name info_tree batch_size')
    exit(1)

pred_data = sys.argv[1]
output_name = sys.argv[2]
info_tree = pkl.load(open(sys.argv[3], 'rb'))
batch_size = int(sys.argv[4])
total_species = len(info_tree)


# species_list = [item[0] for item in sorted(info_tree.items(), key=lambda x: x[1])]
species_list = list(info_tree.keys())


chklines = open(pred_data).readlines()
if len(chklines) <= 1:
    exit(0)

df = pd.read_csv(pred_data,
                 # header=None,
                 sep=',',
                 # index_col=0
                 )
print('df:', df.head())
print('df:', df.shape)
# exit(1)
# df.columns = ['region','species','label','pred']
df.columns = ['region', 'species', 'pred', 'label']
df = df[['region', 'species', 'label', 'pred']]
# df['region'] = df.region.apply(lambda x: x.split('root:')[1])
print('df:', df.head())

uniq_index = df['region'].unique()
# uniq_index = df[0].unique()

df_req = pd.DataFrame(index=uniq_index, columns=species_list+['y'])
# df_req.index = uniq_index
print('df_req:', df_req.shape)

df_group = df.groupby('region')

print('group done')


def mapper(ids):
    def reform(x):
        index = x.name
        # print('index:', index)
        # print('x:', x)
        # sub_df = df[df['hg38_location']==index].T.values
        sub_df = df_group.get_group(index).T.values
        curr_df.loc[index, sub_df[1]] = sub_df[-1]
        curr_df.loc[index, 'y'] = sub_df[-2][0]
        return
    curr_df = pd.DataFrame(index=ids, columns=species_list + ['y'])
    curr_df.swifter.apply(reform,
                                    axis=1)
    # print('curr_df:', curr_df['hg38'])

    return curr_df


list_ids = [uniq_index[i:i + batch_size] for i in range(0, len(uniq_index), batch_size)]

# p = Pool()
# results = p.map(mapper, list_ids)
#
# df_req = pd.concat(results)

df_req = pd.DataFrame()
for item in list_ids:
    get_curr_df = mapper(item)
    df_req = pd.concat([df_req, get_curr_df])

df_req.to_csv(output_name)
