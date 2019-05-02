import os
import pandas as pd
import pickle
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
import scipy

wd = '/Users/lizaharrison/PycharmProjects/Predicting_Papers_v1/Results/'
os.chdir(wd)


# FIGURE 3
pm_web_links = pd.read_pickle('web_data_7.pkl')

links_1 = pm_web_links.set_index('url')
links_1 = links_1.loc[:, 'pmid']

pmid_per_web = links_1.value_counts()
pmid_count_freq = pmid_per_web.value_counts()
pmid_count_freq.sort_index(inplace=True)

links_2 = pm_web_links.set_index('pmid')
links_2 = links_2.loc[:, 'url']

web_per_pmid = links_2.value_counts()
web_count_freq = web_per_pmid.value_counts()
web_count_freq.sort_index(inplace=True)

fig = plt.figure(figsize=(9, 9))
fig.suptitle('Representation of PMIDs in web corpus')
fig, ax = plt.subplots()
ax.scatter(pmid_count_freq.index,
           pmid_count_freq,
           c='#3E0650',
           marker='o',
           s=9,
           zorder=2,
           )
ax.grid(True, zorder=1)
ax.set_yscale('log')
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.ticklabel_format(style='plain')
plt.xlabel('Number of webpages linked to each \nPubMed article')
plt.ylabel('Number of PubMed articles per \nwebpage count')
matplotlib.rcParams.update({'font.size': 14})

plt.show()

#####

# FIGURE 4

plt.close('all')

wd = '/Users/lizaharrison/PycharmProjects/Predicting_Papers_v1/Results/CCA_Results_2'
os.chdir(wd)
all_cca_results = pd.read_csv('ALL_CCA_RESULTS.csv',
                              ).loc[:, ['T-SVD Components',
                                        'CCA Components',
                                        'Median Rank',
                                        ]]
all_cca_results.sort_values(['T-SVD Components',
                            'CCA Components'],
                            inplace=True,
                            )

tsvd_components = [str(value) for value in all_cca_results['T-SVD Components']]
cca_components = [str(value) for value in all_cca_results['CCA Components']]
x_y = zip(tsvd_components,
          cca_components,
          )

median_rank = 100000 / all_cca_results['Median Rank'] * 3

fig1, ax2 = plt.subplots(figsize=(6, 6))

x, y = tsvd_components, cca_components
marker_size = median_rank
colr = median_rank

ax2.scatter(x,
            y,
            c=colr,
            s=marker_size,
            zorder=2,
            cmap='viridis_r',
            )
ax2.set_xlabel('Number of T-SVD components')
ax2.set_ylabel('Number of CCA dimensions')
ax2.grid(True, zorder=1)
matplotlib.rcParams.update({'font.size': 14})

plt.show()


#####

# FIGURE 5
wd = '/Users/lizaharrison/PycharmProjects/Predicting_Papers_v1/Results/CCA_Results_2/Ranks'
os.chdir(wd)

threshold_rank_filename = 'TF-IDF_Thresholds_2_0.85_0_False_0_correct_link_ranks.pkl'
tsvd_rank_filenames = ['TF-IDF_T-SVD_2_1_100_False_0_correct_link_ranks.pkl',
                       'TF-IDF_T-SVD_2_1_200_False_0_correct_link_ranks.pkl',
                       'TF-IDF_T-SVD_2_1_400_False_0_correct_link_ranks.pkl',
                       'TF-IDF_T-SVD_2_1_800_False_0_correct_link_ranks.pkl',
                       'TF-IDF_T-SVD_2_1_1600_False_0_correct_link_ranks.pkl',
                       ]
cca_rank_filenames = ['TF-IDF_T-SVD_2_1_100_True_50_correct_link_ranks.pkl',
                      'TF-IDF_T-SVD_2_1_200_True_100_correct_link_ranks.pkl',
                      'TF-IDF_T-SVD_2_1_400_True_200_correct_link_ranks.pkl',
                      'TF-IDF_T-SVD_2_1_800_True_400_correct_link_ranks.pkl',
                      'TF-IDF_T-SVD_2_1_1600_True_400_correct_link_ranks.pkl',
                      ]

all_rank_files = [threshold_rank_filename]
tsvd_and_cca_ranks = [cca for tsvd in zip(tsvd_rank_filenames, cca_rank_filenames) for cca in tsvd]
all_rank_files.extend(tsvd_and_cca_ranks)
print(all_rank_files)

all_groups_for_vis = ['TF-IDF + THRESHOLDS + NO CCA',
                      'TF-IDF + T-SVD (100) + NO CCA',
                      'TF-IDF + T-SVD (100) + CCA (50)',
                      'TF-IDF + T-SVD (200) + NO CCA',
                      'TF-IDF + T-SVD (200) + CCA (100)',
                      'TF-IDF + T-SVD (400) + NO CCA',
                      'TF-IDF + T-SVD (400) + CCA (200)',
                      'TF-IDF + T-SVD (800) + NO CCA',
                      'TF-IDF + T-SVD (800) + CCA (400)',
                      'TF-IDF + T-SVD (1600) + NO CCA',
                      'TF-IDF + T-SVD (1600) + CCA (400)',
                      ]

all_ranks = []

for filename in all_rank_files:
    print(filename)
    ranks_array = pd.read_pickle(filename)
    all_ranks.append(ranks_array)

final_ranks = dict(zip(all_groups_for_vis, all_ranks))


all_recall_at_n = []

# def cum_sum(binary_threshold_correct_ranks):
for rank_array in final_ranks.values():
    print(rank_array)
    distinct_ranks = np.unique(rank_array)
    rank_counts = pd.Series(rank_array).value_counts()
    rank_counts.sort_index(inplace=True)
    ranks_cumsum = rank_counts.cumsum()
    recall_at_n = ranks_cumsum.apply(lambda x: x / len(rank_array))

    all_recall_at_n.append(recall_at_n)

colours = ['#808080',
           '#91ebce',
           '#015d4f',
           '#76c3f4',
           '#0d136e',
           '#c096cf',
           '#5e004f',
           '#ffb1a1',
           '#970030',
           '#f7c76b',
           '#c95503',
           ]

legend = [Patch(facecolor=colours[0], edgecolor='k', label=all_groups_for_vis[0]),
          Patch(facecolor=colours[1], edgecolor='k', label=all_groups_for_vis[1]),
          Patch(facecolor=colours[2], edgecolor='k', label=all_groups_for_vis[2]),
          Patch(facecolor=colours[3], edgecolor='k', label=all_groups_for_vis[3]),
          Patch(facecolor=colours[4], edgecolor='k', label=all_groups_for_vis[4]),
          Patch(facecolor=colours[5], edgecolor='k', label=all_groups_for_vis[5]),
          Patch(facecolor=colours[6], edgecolor='k', label=all_groups_for_vis[6]),
          Patch(facecolor=colours[7], edgecolor='k', label=all_groups_for_vis[7]),
          Patch(facecolor=colours[8], edgecolor='k', label=all_groups_for_vis[8]),
          Patch(facecolor=colours[9], edgecolor='k', label=all_groups_for_vis[9]),
          Patch(facecolor=colours[10], edgecolor='k', label=all_groups_for_vis[10]),
          ]

fig1, ax1 = plt.subplots(figsize=(12, 7))
i = 0
for data in all_recall_at_n:
    x, y = data.index, data.values
    ax1.scatter(x,
                y,
                c=colours[i],
                label=all_groups_for_vis[i],
                marker='o',
                s=3,
                zorder=5,
                )
    i += 1
matplotlib.rcParams.update({'font.size': 19})
plt.rcParams['legend.handlelength'] = 1
plt.rcParams['legend.handleheight'] = 1.125
ax1.legend(handles=legend, prop={'size': 11})
ax1.grid(True, linewidth=0.75, zorder=0)
ax1.set_xscale('log')
ax1.set_xlabel('Number of PubMed articles analysed (N)')
ax1.set_ylabel('Proportion of known links correctly \nidentified (recall@N)')
ax1.axvline(50, linestyle='-', linewidth=0.9, color='k', zorder=3)
ax1.axvline(1, linestyle='-', linewidth=0.9, color='k', zorder=3)
plt.savefig('Figure_5.png',
            dpi='figure',
            )
plt.show()
