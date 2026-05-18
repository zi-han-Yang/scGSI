import os
import re
import pandas as pd
import scanpy as sc
from seaborn import heatmap
import numpy as np
import matplotlib.pyplot as plt
from eva.eval import *
import seaborn as sns
import math
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score, confusion_matrix


def evaluate_core_lineage(adata, cluster_key, core_clusters, core_edges):
    # 获取子集索引
    all_categories = list(adata.obs[cluster_key].cat.categories)
    subset_indices = [all_categories.index(c) for c in core_clusters if c in all_categories]

    # 提取 PAGA 连通性子矩阵
    paga_conn = adata.uns['paga']['connectivities'].toarray()
    paga_subset = paga_conn[np.ix_(subset_indices, subset_indices)]

    # 构建子集 Ground Truth 矩阵
    n_core = len(core_clusters)
    node_to_idx = {node: i for i, node in enumerate(core_clusters)}
    A_true = np.zeros((n_core, n_core), dtype=int)
    for u, v in core_edges:
        if u in node_to_idx and v in node_to_idx:
            i, j = node_to_idx[u], node_to_idx[v]
            A_true[i, j] = A_true[j, i] = 1

    # 准备评估数据（上三角）
    triu_idx = np.triu_indices(n_core, k=1)
    y_true = A_true[triu_idx]

    # 寻找最佳阈值并计算指标
    best_f1 = 0
    best_t = 0
    results = {}

    for t in np.linspace(0.01, 0.5, 100):
        y_pred = (paga_subset[triu_idx] >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            results = {
                'Best_Threshold': t,
                'F1-score': f1,
                'Jaccard': jaccard_score(y_true, y_pred, zero_division=0),
                'Precision': precision_score(y_true, y_pred, zero_division=0),
                'Recall': recall_score(y_true, y_pred, zero_division=0),
                'y_true': y_true,
                'y_pred': y_pred
            }

    return results


def paga_permutation_test(y_true, y_pred, n_permutations=1000):
    """
    通过打乱预测的边来计算 P-value
    """
    actual_f1 = f1_score(y_true, y_pred)
    null_f1_scores = []

    y_pred_shuffled = y_pred.copy()

    for _ in range(n_permutations):
        np.random.shuffle(y_pred_shuffled)  # 随机打乱预测结果
        null_f1 = f1_score(y_true, y_pred_shuffled)
        null_f1_scores.append(null_f1)

    # 计算 P-value: 随机打乱中 F1 大于等于真实 F1 的比例
    p_value = np.sum(np.array(null_f1_scores) >= actual_f1) / n_permutations

    return actual_f1, null_f1_scores, p_value


def TEA_and_P0_Ablation(vis_dir):
    """
    生成两个并排的柱状图，每个子图内的变体柱子之间完全没有间隙，
    风格参考示例图：简洁网格、特定配色、无顶部/右侧边框。
    """
    save_path_dir = os.path.join(vis_dir, 'TEA_and_P0_Ablation_plot')
    os.makedirs(save_path_dir, exist_ok=True)

    # 五个变体名称
    variants = [
        'scGSI (full)',
        'scGSI-w/o cfc',
        'scGSI-w/o clc',
        'scGSI-w/o pc',
        'scGSI-w/o gc'
    ]

    # 实验数据 (示例值)
    values = np.array([
        [0.762, 0.564, 0.267, 0.668, 0.675],  # TEA_PBMC
        [0.633, 0.338, 0.260, 0.423, 0.429]  # P0BraCor
    ])
    # 颜色方案：参考示例图的深蓝、青绿、嫩绿渐变
    # colors = [
    #     "#FF0000",  # 深蓝 - scMGPF (full)
    #     "#CF3907",  # 浅蓝 - w/o cfc
    #     "#DB538E",  # 绿色 - w/o clc
    #     "#fca2a0",  # 橙色 - w/o pc (示例中类似)
    #     "#FA6540"  # 紫色 - w/o gc
    # ]
    colors = [
        "#E41A1C",  # 深蓝 - scMGPF (full)
        "#FF7F00",  # 浅蓝 - w/o cfc
        "#DB7093",  # 绿色 - w/o clc
        "#FFD000",  # 橙色 - w/o pc (示例中类似)
        "#68ECC2"  # 紫色 - w/o gc
    ]
    # 如果只有3个主要对比，可以使用 colors[:3]，此处假设对应5个变体
    # 若需严格对应图中三色循环，可适当删减或修改 variants 数量

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7, 4.5), sharey=True, dpi=300)
    axes = [ax_left, ax_right]
    titles = ['TEA_PBMC', 'P0BraCor']

    for i, ax in enumerate(axes):
        data = values[i]
        x = np.arange(len(data))

        # 核心修改：width=1.0 确保柱子之间没有缝隙
        # edgecolor='grey' 和 linewidth 提供微小的分割感，类似示例图
        bars = ax.bar(x, data, width=0.8, color=colors[:len(data)],
                      edgecolor='black', linewidth=0.2)
        for bar, score in zip(bars, data):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom',
                    fontsize=9)
        # 设置 X 轴：隐藏刻度线，只保留下方的数据集标签
        ax.set_xticks([])
        ax.set_xlabel(titles[i], fontsize=10, labelpad=10)

        # 样式调整：参考示例图的水平虚线网格
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)  # 确保网格线在柱子后面

    # 左侧图表设置 Y 轴标签
    ax_left.set_ylabel('Per Score', fontsize=11)

    # 添加图例
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[j], ec='black', lw=0.2) for j in range(len(variants))]
    fig.legend(handles, variants, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=3, frameon=False, fontsize=9)

    plt.tight_layout()
    # 为底部的图例留出空间
    plt.subplots_adjust(bottom=0.2)

    save_path = os.path.join(save_path_dir, 'TEA_and_P0_Ablation.png')
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Ablation_plot finished")


def F1_Recall_p_bar(vis_dir):
    """
    生成两个并排的柱状图，每个子图内的变体柱子之间完全没有间隙，
    风格参考示例图：简洁网格、特定配色、无顶部/右侧边框。
    """
    methods = ['JointMDS', 'MMDMA', 'Pamona', 'SCOT', 'UnionCom', 'scTopoGAN', 'scMGCL', 'MultiVI', 'scPairing', 'ScGSI']

    # 实验数据 (示例值)
    values = np.array([
        [0.353, 0.286, 0.339, 0.342, 0.378, 0.398, 0.381, 0.286, 0.343, 0.400],
        [0.818, 0.752, 0.901, 0.632, 0.636, 0.909, 0.727, 0.167, 0.540, 0.910],
        [0.090, 1.000, 0.130, 0.159, 0.086, 0.023, 0.053, 1.000, 0.130, 0.014]
    ])
    # 颜色方案：参考示例图的深蓝、青绿、嫩绿渐变
    # colors = [
    #     "#FF0000",  # 深蓝 - scMGPF (full)
    #     "#CF3907",  # 浅蓝 - w/o cfc
    #     "#DB538E",  # 绿色 - w/o clc
    #     "#fca2a0",  # 橙色 - w/o pc (示例中类似)
    #     "#FA6540"  # 紫色 - w/o gc
    # ]
    colors = [
        "#FF7F00", "#DB7093", "#FFD000", "#68ECC2",
        "#A65628", "#08CC08", "#da13f5", "#A7A4A4", "#1E90FF", "#E41A1C",
        "#ff00dd", "#984EA3", "#999999"]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.5), sharey=False, dpi=300)
    titles = ['F1-score', 'Recall', 'P-value']

    for i, ax in enumerate(axes):
        data = values[i]
        x = np.arange(len(data))

        # 核心修改：width=1.0 确保柱子之间没有缝隙
        # edgecolor='grey' 和 linewidth 提供微小的分割感，类似示例图
        bars = ax.bar(x, data, width=0.8, color=colors[:len(data)],
                      edgecolor='black', linewidth=0.2)
        max_val = np.max(data)
        # 顶部留出 20% 的空间给数值标签
        ax.set_ylim(0, max_val * 1.1)

        for bar, score in zip(bars, data):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + (max_val * 0.02),
                    f'{score:.3f}', ha='center', va='bottom',
                    fontsize=7)
        # 设置 X 轴：隐藏刻度线，只保留下方的数据集标签
        ax.set_xticks([])
        ax.set_title(titles[i], fontsize=11, pad=12)
        # 样式调整：参考示例图的水平虚线网格
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)  # 确保网格线在柱子后面

    # 左侧图表设置 Y 轴标签
    axes[0].set_ylabel('Score', fontsize=11)

    # 添加图例
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[j], ec='black', lw=0.2) for j in range(len(methods))]
    fig.legend(handles, methods, loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=4, frameon=False, fontsize=9)

    plt.tight_layout()
    # 为底部的图例留出空间
    plt.subplots_adjust(bottom=0.22)

    save_path = os.path.join(vis_dir, 'F1_Recall_p_bar.png')
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"F1_Recall_p_bar plot finished")


def scatter_plot(eva_metrics, method_colors, vis_dir):
    categories = eva_metrics.index.tolist()

    method_colors = {key: value for key, value in method_colors.items() if key in categories}
    #eva_metrics_overall = eva_metrics[['FOSCTTM', 'LTA']]

    eva_metrics_overall = eva_metrics[['OMI', 'BCI']]
    # eva_metrics_overall = eva_metrics.iloc[:, :2]
    columns = eva_metrics_overall.columns
    fig, ax = plt.subplots(figsize=(8, 5))

    for category, x, y in zip(categories, eva_metrics_overall[columns[0]], eva_metrics_overall[columns[1]]):
        ax.scatter(x, y, label=category, color=method_colors[category], s=90)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.autoscale(True)

    handles = [plt.Line2D([0], [0],
                          color=method_colors[cat], marker='o', lw=0) for cat in categories]

    legend = ax.legend(handles=handles,
                       labels=categories,
                       loc='center left',
                       bbox_to_anchor=(1.05, 0.5),
                       title="Methods",
                       frameon=False)
    fig.text(0.36, 0.02, ' Omics Mixing Index ', ha='center', fontsize=16)
    fig.text(0.02, 0.5, 'Bio Conservation Index ', va='center', rotation='vertical', fontsize=16)

    #fig.text(0.36, 0.02, 'FOSCTTM', ha='center', fontsize=20)
    #fig.text(0.02, 0.5, 'LTA', va='center', rotation='vertical', fontsize=20)
    plt.tight_layout(rect=[0.05, 0.05, 0.85, 1])
    plt.show()
    #fig.savefig(os.path.join(vis_dir, "FOSCTTM VS LTA.png"),
    #            dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(vis_dir, "OMI VS BCI.png"),
               dpi=600, bbox_inches='tight')
    print(f"scatter_plot finished")


def metrics_plot(eva_metrics, method_colors, vis_dir):
    vis_dir_metrics = vis_dir + '/metrics_Bar_plot'
    if not os.path.exists(vis_dir_metrics):
        os.makedirs(vis_dir_metrics)

    categories = eva_metrics.index.tolist()

    eva_metrics.columns = ['FOSCTTM', 'LTA', 'ARI', 'NMI', 'AMI', 'Omics_mixing', 'Bio_var_conser', 'Accuracy']

    method_colors = {key: value for key, value in method_colors.items() if key in categories}

    # eva_metrics_add = eva_metrics.iloc[:, 2:]
    # columns = eva_metrics_add.columns

    for column in eva_metrics.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        values = eva_metrics[column]
        if np.any(np.isnan(values)):
            ylim_max = 1
        else:
            max_value = values.max()
            ylim_max = math.ceil(max_value / 0.1) * 0.1
        bars = ax.bar(categories, values, color=[method_colors[cat] for cat in categories], alpha=0.8)
        ax.set_title(column)  # 添加标题为列名
        ax.set_ylim(0, ylim_max)
        ax.set_xticks([])  # Remove x-axis labels

        # 为每个柱子添加数值标签
        for bar, value in zip(bars, values):
            if not np.isnan(value):  # 仅为非 NaN 值添加标签
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.3f}',
                        ha='center', va='bottom', fontsize=8)

        handles = [plt.Line2D([0], [0],
                              color=method_colors[cat], lw=4) for cat in categories]

        legend = ax.legend(handles=handles,
                           labels=categories,
                           loc='center left',
                           bbox_to_anchor=(1.05, 0.5),
                           title="Methods",
                           frameon=False)

        fig.text(0.02, 0.5, '', va='center', rotation='vertical', fontsize=14)
        plt.tight_layout(rect=[0.05, 0.05, 0.85, 1])

        # plt.show()
        fig.savefig(os.path.join(vis_dir_metrics, f"{column}_Bar.png"),
                    dpi=600, bbox_inches='tight')
        plt.close(fig)
    print(f"metrics_plot finished")


def umap_plot(adata, omic_colors, cell_type_colors, vis_dir):
    vis_dir_umap = vis_dir + '/UMAP_truth'
    if not os.path.exists(vis_dir_umap):
        os.makedirs(vis_dir_umap)

    obsm_names = adata.obsm_keys()

    for i, method in enumerate(obsm_names):
        sc.pp.neighbors(adata, use_rep=method)
        sc.tl.umap(adata)

        fig, ax = plt.subplots(figsize=(16, 7))
        ax.axis('off')
        ax.set_title(method, fontsize=25, x=0.5, y=0.9)
        inner_ax1 = ax.inset_axes([0.05, 0.1, 0.4, 0.8])
        inner_ax2 = ax.inset_axes([0.55, 0.1, 0.4, 0.8])
        inner_ax1.axis('off')
        inner_ax2.axis('off')

        fig_left = sc.pl.umap(adata,
                              size=80,
                              color='omic_id',
                              palette=omic_colors,
                              title='',
                              ax=inner_ax1,
                              show=False)

        legend = inner_ax1.legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            title='Omic Types',
            frameon=False)
        fig_right = sc.pl.umap(adata,
                               size=100,
                               color="cell_type",
                               palette=cell_type_colors,
                               title='',
                               ax=inner_ax2,
                               show=False)
        legend1 = inner_ax2.legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            title='Cell Types',
            frameon=False)
        plt.tight_layout(pad=3.0)
        plt.show()
        fig.savefig(os.path.join(vis_dir_umap, method + ".png"),
                    dpi=600, bbox_inches='tight')
        plt.close(fig)
    print(f"umap_plot finished")


def umap_clusters_plot(adata, cell_type_colors, eva_dir, vis_dir):
    """
    Generates UMAP visualization of predicted clusters from saved labels.

    Args:
        adata :  the co-embedding NumPy file (e.g., 'embedding.npy').
        eva_dir : Path to the predicted labels file (clu_lables_pred.txt).
        vis_dir : Path to save the plot.
        cell_type_colors
    The function computes UMAP on the embedding, colors by predicted clusters, and optionally by true labels.
    """

    ground_truth = adata.obs['cell_type']
    cell_types = sorted(np.unique(ground_truth))
    n_types = len(cell_types)

    vis_dir_clu_umap = os.path.join(vis_dir, 'Clu_umap')
    if not os.path.exists(vis_dir_clu_umap):
        os.makedirs(vis_dir_clu_umap)
    obsm_names = adata.obsm_keys()

    for i, method in enumerate(obsm_names):
        pred_labels_path = os.path.join(eva_dir, method + "/clu_lables_pred.txt")
        pred_clu_labels = np.loadtxt(pred_labels_path).astype(int)  # Load predicted labels

        pred_clu_labels = [cell_types[label] for label in pred_clu_labels]
        adata.obs['pred_cluster'] = pd.Categorical(pred_clu_labels)
        # Compute UMAP
        sc.pp.neighbors(adata, use_rep=method)
        sc.tl.umap(adata)

        # Plot predicted clusters
        fig, ax = plt.subplots(1, figsize=(8, 6))

        # UMAP colored by predicted clusters
        sc.pl.umap(adata,
                   color='pred_cluster',
                   ax=ax,
                   show=False,
                   frameon=False,
                   palette=cell_type_colors,
                   legend_fontsize=9,
                   legend_fontoutline=1,
                   size=30)
        ax.set_title(f' {method}', fontsize=20, pad=10)

        # Customize legend for cluster-color correspondence
        legend = ax.legend(
            loc='center left',
            bbox_to_anchor=(1.05, 0.5),
            title='Cell Types',
            frameon=False)
        plt.tight_layout()

        # Save
        save_path = os.path.join(vis_dir_clu_umap, method + "_umap_clu.png")
        plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    print(f"umap_clusters_plot finished")


def paga_plot(adata, obsm_names, cell_type_colors, vis_dir, threshold=0.05):
    vis_dir_paga_trajectory = vis_dir + '/PAGA_trajectory' + '_' + str(threshold)
    if not os.path.exists(vis_dir_paga_trajectory):
        os.makedirs(vis_dir_paga_trajectory)

    adata.obs['ground_truth'] = adata.obs['cell_type'].copy()

    if not adata.obs['ground_truth'].dtype.name == 'category':
        adata.obs['ground_truth'] = adata.obs['ground_truth'].astype('category')

    categories = adata.obs['ground_truth'].cat.categories.tolist()

    unique_cell_types_for_colors = np.unique(adata.obs['cell_type'].values)
    color_dict = dict(zip(unique_cell_types_for_colors, cell_type_colors))
    ground_truth_colors = [color_dict[cat] for cat in categories]

    adata.uns['ground_truth_colors'] = ground_truth_colors

    for method in obsm_names:

        sc.pp.neighbors(adata, use_rep=method)
        sc.tl.umap(adata)
        sc.tl.paga(adata, groups='ground_truth')

        fig, ax = plt.subplots(figsize=(22, 15))
        sc.pl.paga(adata,
                   threshold=threshold,
                   labels=None,
                   show=False,
                   ax=ax,
                   node_size_scale=10,
                   edge_width_scale=2,
                   node_size_power=0.5,
                   frameon=False)

        ax.axis('off')
        ax.set_title(f'{method}',
                     fontsize=60, x=0.5, y=0.95, ha='center')

        for artist in ax.get_children():
            if isinstance(artist, plt.Text):
                artist.set_visible(5)

        handles = [plt.Line2D([0], [0],
                              marker='o',
                              color='w',
                              label=cell_type,
                              markersize=10,
                              markerfacecolor=color)
                   for cell_type, color in zip(categories, cell_type_colors)]

        legend = ax.legend(handles=handles,
                           loc='center left',
                           bbox_to_anchor=(1, 0.5),
                           title='Cell Types')

        legend.get_frame().set_linewidth(0)
        plt.setp(legend.get_texts(), fontsize=13)
        legend.get_title().set_fontsize(20)

        plt.tight_layout()
        fig.subplots_adjust(right=0.85)
        # plt.show()
        fig.savefig(os.path.join(vis_dir_paga_trajectory, method + ".png"),
                    dpi=600, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    print(f"paga_plot finished")


def plot_shared_top_genes_expression(adata, RNA_data, ATAC_data, vis_dir, top_n=5, title='Shared Top Genes Expression'):
    """
    Plot expression of the top shared genes for RNA and ATAC.
    Each gene generates 2 separate figures (RNA and ATAC), saved as
    '{title} - RNA - {gene}.png' and '{title} - ATAC - {gene}.png'.

    Expression values are adjusted to eliminate negative values (shifted so minimum becomes 0).
    Each figure includes:
      - Scatter plot of UMAP coordinates colored by expression, with a visible frame (box).
      - A side horizontal bar showing mean expression, with bar height matching the scatter frame height.
      - Title set to only the gene name.

    Args:
        adata: combined AnnData containing both RNA and ATAC cells.
        RNA_data: raw RNA AnnData.
        ATAC_data: raw ATAC AnnData.
        vis_dir: directory to save the figures.
        top_n: number of shared top genes to plot (default: 5).
        title: overall title prefix for the saved filenames.
    """
    vis_dir_genes = os.path.join(vis_dir, 'shared_gene_expression')
    os.makedirs(vis_dir_genes, exist_ok=True)

    if 'omic_id' not in adata.obs or 'cell_type' not in adata.obs:
        raise ValueError('adata must contain obs columns "omic_id" and "cell_type" for modality separation.')

    # Helper function to adjust expression values to reduce negative values
    def adjust_expression(vals):
        vals = np.asarray(vals).ravel()
        min_val = np.nanmin(vals)
        if min_val < 0:
            vals = vals - min_val  # shift so min becomes 0
        return vals

    shared_genes = sorted(set(RNA_data.var_names).intersection(set(ATAC_data.var_names)))
    if len(shared_genes) == 0:
        raise ValueError('No shared genes found between RNA and ATAC data.')

        # Use original (unadjusted) values for gene ranking
    gene_scores = {}
    for gene in shared_genes:
        rna_vals = np.asarray(RNA_data[:, gene].X).ravel()
        atac_vals = np.asarray(ATAC_data[:, gene].X).ravel()
        gene_scores[gene] = np.nanmean(rna_vals) + np.nanmean(atac_vals)

    top_genes = sorted(gene_scores, key=gene_scores.get, reverse=True)[:min(top_n, len(gene_scores))]

    if 'X_umap' not in adata.obsm:
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)

    umap_coords = adata.obsm['X_umap']
    rna_mask = adata.obs['omic_id'].values == 'RNA'
    atac_mask = adata.obs['omic_id'].values == 'ATAC'
    rna_umap = umap_coords[rna_mask]
    atac_umap = umap_coords[atac_mask]

        # Helper: create a scatter plot with a frame and a side expression distribution bar matching the frame height
    def _plot_gene_figure(umap_xy, expr_vals, gene_name, modality_label, save_dir, title_prefix):
        expr_vals = adjust_expression(np.asarray(expr_vals).ravel())

        fig = plt.figure(figsize=(6.5, 4.5), dpi=200)

        # [left, bottom, width, height] — all axes share the same height
        ax_cbar = fig.add_axes([0.08, 0.15, 0.01, 0.75])   # colorbar on the left
        ax_scatter = fig.add_axes([0.10, 0.15, 0.64, 0.75])  # scatter in the middle
        ax_bar = fig.add_axes([0.75, 0.15, 0.06, 0.75])    # expression bar on the right

        # Scatter plot with frame (box)
        pts = ax_scatter.scatter(umap_xy[:, 0], umap_xy[:, 1], c=expr_vals,
                                 cmap='viridis', s=15, edgecolors='none')
        ax_scatter.set_title(gene_name, fontsize=12)
        ax_scatter.set_xticks([])
        ax_scatter.set_yticks([])
        # Draw black frame around scatter
        for spine in ax_scatter.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)

        # Side bar: histogram of expression values as horizontal bars, height matches frame
        counts, bin_edges = np.histogram(expr_vals, bins=20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # Normalize bar width to fill the full axes height
        bar_height = 1.0 / len(counts)
        for i, (c, bc) in enumerate(zip(counts, bin_centers)):
            ax_bar.barh(i / len(counts), c, height=bar_height,
                        color=plt.cm.viridis(bc / max(bin_centers.max(), 1e-10)),
                        edgecolor='none')
        ax_bar.set_xlim(0, max(counts) * 1.1 if len(counts) > 0 else 1)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_yticks([])
        ax_bar.set_xlabel('Count', fontsize=8)
        ax_bar.tick_params(labelsize=7)
        # Draw black frame around bar
        for spine in ax_bar.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)



                # Colorbar — use pre-positioned ax_cbar on the left
        cbar = fig.colorbar(pts, cax=ax_cbar)
        cbar.ax.tick_params(labelsize=7)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')

        safe_gene = re.sub(r'[^\w\-_\. ]', '_', gene_name)
        save_path = os.path.join(save_dir, f'{title_prefix} - {modality_label} - {safe_gene}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved: {save_path}")

    # Save each gene as separate RNA and ATAC figures
    for gene in top_genes:
        rna_vals = np.asarray(RNA_data[:, gene].X)
        atac_vals = np.asarray(ATAC_data[:, gene].X)
        safe_gene = re.sub(r'[^\w\-_\. ]', '_', gene)

        # RNA figure
        _plot_gene_figure(rna_umap, rna_vals, gene, 'RNA', vis_dir_genes, title)

        # ATAC figure
        _plot_gene_figure(atac_umap, atac_vals, gene, 'ATAC', vis_dir_genes, title)

    print(f"plot_shared_top_genes_expression finished: {len(top_genes)} gene(s) saved, {len(top_genes) * 2} figure(s) in total.")



def clu_heatmap_plot(adata, obsm_names, CM, accuracy, vis_dir, keys='clu_heatmap_plot'):
    vis_dir_confusion = os.path.join(vis_dir, keys)
    if not os.path.exists(vis_dir_confusion):
        os.makedirs(vis_dir_confusion)

    ground_truth = adata.obs['cell_type']
    labels = sorted(np.unique(ground_truth))
    n_classes = len(labels)

    fig_size = max(8, n_classes * 0.5)
    cmap = 'Reds'

    for method in obsm_names:
        cm = CM[method].astype(float)
        cm = cm / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))

        ax = heatmap(
            cm,
            ax=ax,
            xticklabels=labels,
            yticklabels=labels,
            cmap=cmap,
            annot=True,
            fmt='.3f',
            cbar_kws={'shrink': 0.71, 'label': 'Normalized clustering Prediction Probability'},
            square=True
        )

        for text in ax.texts:
            if float(text.get_text()) == 0.0:
                text.set_text('')

        ax.set_title(
            f'{method}\n Clustering Accuracy: {accuracy[method]:.2f}',
            fontsize=24,
            pad=20,
        )

        ax.set_xticklabels(ax.get_xticklabels(),
                           rotation=45,
                           ha='right',
                           fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(),
                           rotation=0,
                           fontsize=10)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=9)

        plt.tight_layout()

        save_path = os.path.join(vis_dir_confusion, f'{method}.png')
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
    print(f"clu_heatmap_plot finished")


def classified_confusion_matrices(adata, eva_dir, vis_dir, clu_rna):
    """
    Reads true and predicted labels from eva_dir for each method, computes confusion matrices,
    and saves heatmaps to vis_dir.

    Assumes:
    - eva_dir contains subdirectories named after methods.
    - Each method subdirectory has {pred_file_pattern} (e.g., 'pred_labels.txt').
    - True labels are shared in eva_dir/{true_file} (e.g., 'true_labels.txt').

    Args:
        eva_dir (str): Path to evaluation directory.
        vis_dir (str): Path to visualization directory .
        adata:
        clu_rna:
    """
    pred_file_pattern = 'rna_Classifier_label_predict.txt'
    # Create vis_dir if not exists
    vis_dir_confusion = vis_dir + '/' + 'fenlei_confusion_matrix'
    os.makedirs(vis_dir_confusion, exist_ok=True)
    # Get methods (subdirectories in eva_dir)
    methods = [d for d in os.listdir(eva_dir) if os.path.isdir(os.path.join(eva_dir, d))]
    # Load shared true labels
    true_labels = clu_rna
    ground_truth = adata.obs['cell_type']
    labels = sorted(np.unique(ground_truth))
    for method in methods:

        pred_path = os.path.join(eva_dir, method, pred_file_pattern)
        if not os.path.exists(pred_path):
            print(f"Predicted labels not found for {method}: {pred_path}. Skipping.")
            continue

        # Load predicted labels
        pred_labels = np.loadtxt(pred_path)

        # Compute accuracy (LTA)
        correct = np.sum(pred_labels == true_labels)
        accuracy = correct / len(true_labels)

        # Compute confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)

        # Visualize and save
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm,
                    xticklabels=labels,
                    yticklabels=labels,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    cbar=False)

        plt.title(f'{method}\n LTA of True vs Predicted Labels ({accuracy:.3f})', fontsize=22, pad=20)

        save_path = os.path.join(vis_dir_confusion, method + ".png")
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
    print(f"classified_confusion_matrices finished")


def radar_plot(eva_metrics, vis_dir, colors):
    # Input validation
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    if eva_metrics.empty:
        raise ValueError("eva_metrics must be a non-empty DataFrame")

    # Get method names (index) and metric names (columns)
    methods = eva_metrics.index.tolist()
    metrics = eva_metrics.columns.tolist()

    # Normalize FOSCTTM: inverse, log scale, and standardize
    df_processed = eva_metrics.copy()
    # df_processed = df_processed.drop('raw_data', errors='ignore')

    if 'FOSCTTM' in metrics:
        df_processed['FOSCTTM'] = 1 - df_processed['FOSCTTM'].astype(float)

    # Radar chart setup
    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Initialize figure with journal-quality settings
    fig, ax = plt.subplots(figsize=(12, 12),
                           subplot_kw=dict(polar=True), dpi=600)
    max_value = df_processed.max().max()
    min_value = df_processed.min().min()
    if max_value == min_value:
        max_value += 0.1
    ax.set_rlim(0.0, 1.0)  # Explicitly set rlim from 0 to 1 for center-to-edge scaling

    # Draw radar polygons without filling
    for idx, method in enumerate(methods):
        values = df_processed.loc[method].values.flatten().tolist()
        values += values[:1]

        ax.plot(angles, values,
                linewidth=2.5,
                linestyle='solid',
                label=method,
                color=colors[method],
                zorder=10)

        # Customize axes and labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])  # Hide default tick labels to avoid overlap

        # Manually place metric labels outside the chart to avoid intersection, upright (no rotation)
        label_radius = 1.12  # Closer to the 1.0 rim
        for i, metric in enumerate(metrics):
            theta = angles[i]
            ax.text(theta, label_radius, metric,
                    ha='center', va='bottom',
                    fontsize=17,
                    color='black',
                    rotation=0,  # No rotation, upright text
                    rotation_mode='anchor')

    # Enhance radial grid with custom scale annotations
    r_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_rgrids([0.0] + r_levels, labels=None)  # Disable default labels
    ax.set_yticklabels([])  # Explicitly hide any remaining r tick labels

    # Small angle offset to place labels beside the radial lines
    delta_theta = 0.05  # Smaller offset in radians for closer placement beside lines

    for i, metric in enumerate(metrics):
        theta = angles[i]
        is_reverse = (metric == 'FOSCTTM')
        # Alternate offset direction for better spacing (left/right of line)
        sign = 1 if i % 2 == 0 else -1
        for r_level in r_levels:
            offset_r = 0.015 if r_level < 1.0 else -0.02
            display_r = r_level + offset_r
            if is_reverse:
                label_val = 1.0 - r_level
            else:
                label_val = r_level
            ax.text(theta + sign * delta_theta, display_r, f'{label_val:.1f}',
                    ha='center', va='center',
                    fontsize=15,
                    color='black',
                    alpha=0.9,
                    rotation=0,
                    rotation_mode='anchor')

    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.9, color='black')

    # Improve legend and background
    ax.spines['polar'].set_visible(False)
    ax.set_facecolor('#f5f5f5')
    fig.patch.set_facecolor('white')
    plt.legend(loc='upper right',
               bbox_to_anchor=(1.2, 1.1),
               fontsize=12,
               facecolor='white',
               edgecolor='black')

    # Save with high resolution
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'radar_plot_ablation.png'),
                dpi=600, bbox_inches='tight', facecolor='white')
    #plt.show()
    plt.close()

    print(f"radar_plot finished")


def plot_permutation_results(actual_f1, null_f1_scores, p_value, vis_dir, obsm_names):

    # 1. 环境配置：设置全局学术绘图风格
    sns.set_theme(style="white")  # 清洁背景
    plt.rcParams['font.family'] = 'Arial'  # 期刊常用字体
    plt.rcParams['pdf.fonttype'] = 42  # 确保导出PDF可编辑

    vis_dir_metrics = os.path.join(vis_dir, 'Permutation_Test')
    os.makedirs(vis_dir_metrics, exist_ok=True)
    for method in obsm_names:
        fig, ax = plt.subplots(figsize=(6, 4.5))  # 黄金比例

        # 2. 绘制零假设分布 (Null Distribution)
        # 使用 hist + kde，并调整色彩为学术灰蓝调
        sns.histplot(null_f1_scores, bins=30, kde=True,
                     color='#B0BEC5',  # 学术浅灰色
                     edgecolor='white', linewidth=0.5,
                     alpha=0.8, ax=ax, label='Null distribution')

        null_mean = np.mean(null_f1_scores)
        null_std = np.std(null_f1_scores)
        ax.axvline(null_mean, color='#475569', linestyle='--', linewidth=1.5,
                   label=f'Null mean = {null_mean:.3f} ± {null_std:.3f}')

        # 3. 绘制观察到的真实值 (Observed Value)
        ax.axvline(actual_f1, color='#D32F2F',  # 经典学术红
                   linestyle='--', linewidth=2,
                   label=f'Observed F1 = {actual_f1:.3f}')

        # 4. 动态显著性标注 (Significance Star)
        # 根据 P-value 自动分配星号
        if p_value < 0.001:
            sig = '***'
        elif p_value < 0.01:
            sig = '**'
        elif p_value < 0.05:
            sig = '*'
        else:
            sig = 'n.s.'

        # 在红线上方添加标注
        y_max = ax.get_ylim()[1]
        ax.text(actual_f1, y_max * 0.9, f'{sig}\n$P$ = {p_value:.4f}',
                color='#D32F2F', weight='bold', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # 5. 精细化坐标轴与标签
        ax.set_title(f'Trajectory Consistency Analysis ({method})',
                     fontsize=12, pad=15, weight='bold')
        ax.set_xlabel('F1-score (Random Permutation)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)

        # 移除顶部和右侧的边框 (Spines)
        sns.despine()

        # 6. 图例美化
        ax.legend(frameon=False, loc='upper left', fontsize=9)

        # 7. 布局微调与保存
        plt.tight_layout()
        save_filename = os.path.join(vis_dir_metrics, f"{method}.png")  # 推荐保存为PDF
        plt.savefig(save_filename, dpi=600, bbox_inches='tight')
        plt.savefig(save_filename.replace('.pdf', '.png'), dpi=600, bbox_inches='tight')

        plt.show()
        plt.close()


def purity_box_plot(adata, obsm_names, vis_dir, cluster_labels_np, colors):
    vis_dir_purity = os.path.join(vis_dir, 'Purity_box')
    if not os.path.exists(vis_dir_purity):
        os.makedirs(vis_dir_purity)
    # obsm_names = obsm_names.drop('raw_data', errors='ignore')
    # k 值范围
    k_values = list(range(0, 21, 5))
    k_values.append(1)
    # 计算每个方法在不同 k 下的纯度
    purity_data = []
    for method in obsm_names:
        purities = [knn_purity_score(adata.obsm[method], cluster_labels_np, k) for k in k_values]
        for k, purity in zip(k_values, purities):
            purity_data.append({'Method': method, 'k': k, 'Purity': purity})

    # 转换为 DataFrame
    df = pd.DataFrame(purity_data)

    # 设置专业样式
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))

    palette = dict(zip(obsm_names, colors[:len(obsm_names)]))
    # 绘制箱型图
    # sns.boxplot(data=df,
    #             x='Method',
    #             y='Purity',
    #             palette=palette,
    #             width=0.4,
    #             fliersize=4,
    #             linewidth=1)
    # 绘制折线图
    sns.lineplot(data=df,
                 x='k',
                 y='Purity',
                 hue='Method',
                 palette=palette,
                 linewidth=2.0)

    # 设置标题和标签
    plt.title('Purity Score ',
              fontsize=20, pad=20)
    plt.ylabel('kNN Purity Score', fontsize=18)
    plt.xlabel('Number of Nearest Neighbors', fontsize=18)

    # X 轴设置
    plt.xlim(0, 20)
    plt.xticks([0, 5, 10, 15, 20], fontsize=12)

    # Y 轴设置
    plt.ylim(0, 1.1)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12)

    # 图例设置
    plt.legend(title='Methods',
               title_fontsize=13,
               loc='upper right',
               bbox_to_anchor=(1.2, 1.1),
               fontsize=12,
               facecolor='white',
               edgecolor='black')

    # 修改 x 轴和 y 轴边框线为黑线
    ax = plt.gca()
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    # 调整布局
    plt.tight_layout()
    # 保存高分辨率图像
    save_path = os.path.join(vis_dir_purity, "purity_lineplot.png")
    plt.savefig(save_path,
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"purity_box_plot finished")


def total_performance_plot(eva_metrics, method_colors, vis_dir):
    """
    Computes the overall performance score for each method by averaging all metrics except the first one
    (without normalization), then generates a bar plot.

    Args:
        eva_metrics (pd.DataFrame): DataFrame with methods as index, metrics as columns.
        method_colors (dict): Dict mapping method names to colors.
        vis_dir (str): Directory to save the plot.
    """
    # Skip 'raw_data' if present
    # eva_metrics = eva_metrics.drop('raw_data', errors='ignore')

    # Compute overall performance as row-wise mean, excluding the first metric (raw values)
    first_metric = eva_metrics.columns[0]
    eva_metrics['total_performance'] = eva_metrics.drop(columns=[first_metric]).mean(axis=1)

    # Prepare data for plotting
    methods = eva_metrics.index.tolist()
    total_scores = eva_metrics.drop(columns=[first_metric]).mean(axis=1).values

    # Create bar plot
    fig, ax = plt.subplots(figsize=(7, 5), dpi=600)
    bars = ax.bar(range(len(methods)), total_scores, color=[method_colors[method] for method in methods], alpha=0.8)

    # Customize plot

    ax.set_title(f'\nTotal Performance', fontsize=20, pad=20)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([])

    # Add legend with methods and colors
    handles = [plt.Line2D([0], [0],
                          color=method_colors[method], lw=4) for method in methods]
    legend = ax.legend(handles,
                       methods,
                       loc='center left',
                       bbox_to_anchor=(1, 0.5),
                       title="Methods",
                       frameon=False)

    # Add value labels on bars
    for bar, score in zip(bars, total_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)

    # Save plot
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    save_path = os.path.join(vis_dir, f'total_performance.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"total_performance_plot finished")


def mix_and_bio_plot(mix_and_bio_metric, method_colors, vis_dir):
    """
    Generates grouped bar plots for omics mixing and bio var conservation metrics.

    Args:
        mix_and_bio_metric (pd.DataFrame): DataFrame with metrics as columns and methods as index.
        method_colors (dict): Dict of method names to colors for bars.
        vis_dir (str): Directory to save the plot.
    """
    # Create directory if not exists
    mix_bio_path = os.path.join(vis_dir, 'mix_and_bio_Bar_plot')
    os.makedirs(mix_bio_path, exist_ok=True)

    # Select only the 6 specific metrics (columns)
    metric_names = ['NOS', 'GC', 'SAS', 'ASW', 'PS', 'MAP']
    mix_and_bio_metric = mix_and_bio_metric[metric_names]  # Slice to the exact 6 columns

    # Methods (index)
    methods = mix_and_bio_metric.index.tolist()

    # Colors for methods
    colors = [method_colors[method] for method in methods]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=600)

    # Left subplot: Omics Mixing Metrics (first 3 columns: NOC, GC, SAS)
    mixing_metrics = mix_and_bio_metric[['NOS', 'GC', 'SAS']]  # Select first 3 columns, rows methods
    x_pos = np.arange(len(mixing_metrics.columns))  # Positions for metrics (3,)
    width = 0.8 / len(methods)  # Adjusted width for number of methods

    for i, method in enumerate(methods):
        values = mixing_metrics.loc[method].values  # Row for method, shape (3,)
        bars = ax1.bar(x_pos + i * width, values, width, label=method, color=colors[i], alpha=0.9)

    ax1.set_xlabel('', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Omics Mixing Index', fontsize=18, pad=20)
    ax1.set_xticks(x_pos + width * (len(methods) - 1) / 2)
    ax1.set_xticklabels(['NOS', 'GC', 'SAS'], fontsize=10)

    # Set y-limit slightly higher to fit labels
    max_y1 = max(mixing_metrics.max().max(), 0) * 1.05
    ax1.set_ylim(0, max_y1)

    # Right subplot: Bio Var Conservation Metrics (last 3 columns: ASW, PS, MAP)
    bio_metrics = mix_and_bio_metric[['ASW', 'PS', 'MAP']]  # Select last 3 columns, rows methods
    for i, method in enumerate(methods):
        values = bio_metrics.loc[method].values  # Row for method, shape (3,)
        bars = ax2.bar(x_pos + i * width, values, width, label=method, color=colors[i], alpha=0.9)

    ax2.set_xlabel('', fontsize=12, fontweight='bold')
    ax2.set_ylabel('', fontsize=12)
    ax2.set_title('Bio Conservation Index', fontsize=18, pad=20)
    ax2.set_xticks(x_pos + width * (len(methods) - 1) / 2)
    ax2.set_xticklabels(['ASW', 'PS', 'MAP'], fontsize=10)
    handles = [plt.Line2D([0], [0],
                          color=method_colors[cat], lw=3) for cat in methods]

    ax2.legend(handles=handles,
               labels=methods,
               bbox_to_anchor=(1.05, 1),
               loc='upper left',
               title="Methods",
               frameon=False,
               fontsize=9)

    # Set y-limit slightly higher to fit labels
    max_y2 = max(bio_metrics.max().max(), 0) * 1.05
    ax2.set_ylim(0, max_y2)

    # Overall layout and save
    plt.tight_layout()
    save_path = os.path.join(mix_bio_path, 'mix_bio_bar.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"mix_and_bio_abr_plot finished")

# Example usage:
# visualize_umap_clusters('path/to/embedding.npy', 'path/to/clu_lables_pred.txt',
#                         true_labels_path='path/to/true_labels.txt',
#                         dataset_name='PBMC10k', method_name='scMGFP')
# Example usage:
# mix_and_bio_metrics = pd.read_csv(eva_dir + '/mix_and_bio_metrics_' + dataset_name + '.csv', index_col=0, header=0)
# mix_and_bio_plot(mix_and_bio_metrics, method_colors, vis_dir)

# Example usage:
# plot_mix_and_bio_metrics('eva_dir', 'vis_dir', 'PBMC10k', method_colors)
# Usage example:
# eva_dir = r"E:\experiment\DaOT\eva\AdBraCor"
# vis_dir = r"E:\experiment\DaOT\vis\AdBraCor"
# generate_confusion_matrices(eva_dir, vis_dir)
# def confusion_plot(adata, obsm_names, CM, accuracy, vis_dir, keys='confusion_plot'):
#     vis_dir_confusion = vis_dir + '/' + keys
#     if not os.path.exists(vis_dir_confusion):
#         os.makedirs(vis_dir_confusion)
#
#     ground_truth = adata.obs['cell_type']
#
#     for method in obsm_names:
#         cm = CM[method]
#         cm = cm / cm.sum(axis=1, keepdims=True)
#         fig, ax = plt.subplots(1, 1, figsize=(30, 20))
#         labels = np.unique(ground_truth)
#         ax.set_title(method, fontsize=20, x=0.5, y=0.9)
#         ax = heatmap(cm, ax=ax, xticklabels=labels, yticklabels=labels, cmap='Reds')  # 'BuGn'
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=20)
#         ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=20)
#         ax.set_title(f'{method} (Accuracy: {accuracy[method]:.2f})', fontsize=30)
#         cbar = plt.gca().collections[0].colorbar
#         cbar.ax.tick_params(labelsize=40)
#         # plt.show()
#         fig.savefig(os.path.join(vis_dir_confusion, method + ".png"), dpi=300, bbox_inches='tight')
#         plt.close(fig)


# def radar_plot1(eva_metrics, vis_dir):
#     # Input validation
#     if not os.path.exists(vis_dir):
#         os.makedirs(vis_dir)
#     if eva_metrics.empty:
#         raise ValueError("eva_metrics must be a non-empty DataFrame")
#
#     # Get method names (index) and metric names (columns)
#     methods = eva_metrics.index.tolist()
#     metrics = eva_metrics.columns.tolist()
#
#     # Normalize FOSCTTM: inverse, log scale, and standardize
#     df_processed = eva_metrics.copy()
#     if 'FOSCTTM' in metrics:
#         df_processed['FOSCTTM'] = 1 / df_processed['FOSCTTM']
#         df_processed['FOSCTTM'] = np.log1p(df_processed['FOSCTTM'])
#         foscttm_values = df_processed['FOSCTTM'].astype(float)
#         if (foscttm_values <= 0).any():
#             raise ValueError("FOSCTTM values must be positive")
#         df_processed['FOSCTTM'] = 1 / foscttm_values
#         df_processed['FOSCTTM'] = np.log1p(df_processed['FOSCTTM'])
#         foscttm_min, foscttm_max = foscttm_values.min(), foscttm_values.max()
#         if foscttm_max != foscttm_min:
#             df_processed['FOSCTTM'] = (foscttm_values - foscttm_min) / (foscttm_max - foscttm_min)
#         else:
#             df_processed['FOSCTTM'] = 0.5
#
#     # Radar chart setup
#     N = len(metrics)
#     angles = [n / float(N) * 2 * np.pi for n in range(N)]
#     angles += angles[:1]
#
#     # Initialize figure with journal-quality settings
#     fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True), dpi=300)
#     max_value = df_processed.max().max()
#     min_value = df_processed.min().min()
#     if max_value == min_value:
#         max_value += 0.1
#     ax.set_rlim(min_value - 0.1 * (max_value - min_value), max_value + 0.1 * (max_value - min_value))
#
#     # Draw radar polygons with enhanced styling
#     colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
#
#     # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
#     #           '#17becf']
#     for idx, method in enumerate(methods):
#         values = df_processed.loc[method].values.flatten().tolist()
#         values += values[:1]
#         ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=method, color=colors[idx], zorder=10)
#         ax.fill(angles, values, color=colors[idx], alpha=0.15, zorder=5)
#
#     # Customize axes and labels
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(metrics, fontsize=12, fontweight='bold', rotation_mode='anchor', ha='center')
#     ax.set_title('Performance Comparison of Methods Across Metrics', size=18, pad=30, fontweight='bold',
#                  fontfamily='serif')
#
#     # Enhance radial grid
#     ax.set_rgrids([0.2, 0.4, 0.6, 0.8], labels=['0.2', '0.4', '0.6', '0.8'], angle=0, fontsize=10, color='black',
#                   alpha=0.9)
#     ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.9, color='gray')
#
#     # Add statistical annotations
#     for i, (angle, metric) in enumerate(zip(angles[:-1], metrics)):
#         value = df_processed[metric].mean()
#         ax.text(angle, value + 0.05, f'{value:.2f}', ha='center', va='bottom', fontsize=8, color='black')
#
#     # Improve legend and background
#     ax.spines['polar'].set_visible(False)
#     ax.set_facecolor('#f5f5f5')
#     fig.patch.set_facecolor('white')
#     plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=12, frameon=True, facecolor='white',
#                edgecolor='black')
#
#     # Save with high resolution
#     plt.tight_layout()
#     plt.savefig(os.path.join(vis_dir, 'radar_plot.png'), dpi=600, bbox_inches='tight', facecolor='white')
#     plt.show()


# def paga_plot_colors1(adata, obsm_names, cell_type_colors, vis_dir, threshold=0.05):
#     vis_dir_paga_trajectory = vis_dir + '/PAGA_trajectory' + '_' + str(threshold)
#     if not os.path.exists(vis_dir_paga_trajectory):
#         os.makedirs(vis_dir_paga_trajectory)
#
#     adata.obs['ground_truth'] = adata.obs['cell_type'].copy()
#
#     # 转换为 categorical 以访问 .cat.categories
#     if not adata.obs['ground_truth'].dtype.name == 'category':
#         adata.obs['ground_truth'] = adata.obs['ground_truth'].astype('category')
#
#     categories = adata.obs['ground_truth'].cat.categories.tolist()  # 获取分组的实际类别顺序
#
#     # 使用 np.unique 获取排序后的唯一细胞类型（匹配提供的 cell_type_colors 顺序）
#     unique_cell_types_for_colors = np.unique(adata.obs['cell_type'].values)
#     color_dict = dict(zip(unique_cell_types_for_colors, cell_type_colors))
#     ground_truth_colors = [color_dict[cat] for cat in categories]
#
#     # 设置正确的颜色键供 PAGA 使用
#     adata.uns['ground_truth_colors'] = ground_truth_colors
#
#     for method in obsm_names:
#         sc.pp.neighbors(adata, use_rep=method)
#         sc.tl.umap(adata)
#         sc.tl.paga(adata, groups='ground_truth')
#
#         fig, ax = plt.subplots(figsize=(20, 12))  # 调整大小以更好地适应图例和标签
#         sc.pl.paga(adata, threshold=threshold, labels=None, show=False, ax=ax,
#                    node_size_scale=10, edge_width_scale=1.5,  # 略微增大节点，减小边宽以突出结构
#                    node_size_power=0.5,  # 调整缩放以改善视觉平衡
#                    frameon=False)  # 移除边框以更简洁
#         ax.axis('off')
#         ax.set_title(f'PAGA Trajectory: {method}\n(Threshold = {threshold})',
#                      fontsize=18, x=0.5, y=0.95, ha='center', fontweight='bold')  # 改进标题：添加阈值、更居中、加粗
#
#         for artist in ax.get_children():
#             if isinstance(artist, plt.Text) and artist.get_text() in categories:
#                 artist.set_fontsize(5)
#
#         # 图例使用 categories 和 ground_truth_colors，确保顺序一致，并改进样式
#         handles = [plt.Line2D([0], [0], marker='o', color='w', label=cell_type,
#                               markersize=8, markerfacecolor=color, markeredgecolor='black', markeredgewidth=0.5)
#                    for cell_type, color in zip(categories, ground_truth_colors)]
#         legend = ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5),
#                            title='Cell Types', fancybox=False, shadow=False,
#                            title_fontsize=16, fontsize=12, ncol=1)  # 改进图例：添加边框、调整字体、无阴影、多列可选但这里单列
#         legend.get_frame().set_linewidth(0.5)
#         legend.get_frame().set_edgecolor('gray')
#         legend.get_title().set_fontweight('bold')
#
#         plt.tight_layout()  # 自动调整布局以防止重叠
#         fig.subplots_adjust(right=0.85)  # 为图例留出更多空间
#
#         # plt.show()
#         fig.savefig(os.path.join(vis_dir_paga_trajectory, method + "1.png"), dpi=300, bbox_inches="tight",
#                     pad_inches=0.1)
#         plt.close(fig)
