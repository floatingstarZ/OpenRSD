import numpy as np
from commonlibs.common_tools import *
import os
import torch
import matplotlib.pyplot as plt
import random
from matplotlib import colors  # 注意！为了调整“色盘”，需要导入colors
from sklearn.manifold import TSNE
from tqdm import tqdm
import colorsys
from pathlib import Path

def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    if num == 0:
        return []
    step = 360.0 / num
    while i < 360:
        if len(hls_colors) == num:
            break
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors

def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors

def norm_colors(colors):
    """
    (a, b, c) -> ele in [0, 1]
    :param colors: [(a, b, c)]
    :return:
    """
    r_c = []
    def norm(c):
        c = np.array(c)
        if np.max(c) == 0:
            return list(c)
        else:
            return list(c/np.max(c))

    for c in colors:
        c = norm(c)
        r_c.append(c)
    return r_c


def plot_embedding_2d(X, labels, save_pth):
    """Plot an embedding X with the class label y colored by the domain d."""
    # x_min, x_max = np.min(X, 0), np.max(X, 0)
    # X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.rcParams['figure.dpi'] = 600
    plt.figure(figsize=(32, 32))

    # Start from zero
    max_labels = max(set(labels))

    colors = ncolors(max_labels + 1)
    colors = norm_colors(colors)

    plt.xlim(np.min(X[:, 0]) - 0.1, np.max(X[:, 0]) + 0.1)
    plt.ylim(np.min(X[:, 1]) - 0.1, np.max(X[:, 1]) + 0.1)
    for i, c in tqdm(list(enumerate(colors))):
        x = X[labels == i]
        for j in range(x.shape[0]):
            # print(j, c, ys[i], x[j])
            # plot colored number
            plt.text(x[j, 0], x[j, 1], str(i),
                     color=c,
                     fontdict={'weight': 'bold', 'size': 2})

    plt.savefig(save_pth)

import faiss
import time
import numpy as np
from PIL import Image
from PIL import ImageFile
from scipy.sparse import csr_matrix, find
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from copy import deepcopy


def preprocess_features(npdata, pca=384):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    pca_feat = mat.apply_py(npdata)

    pca_feat = deepcopy(pca_feat)
    b = faiss.vector_to_array(mat.b)
    A = faiss.vector_to_array(mat.A).reshape(mat.d_out, mat.d_in)
    # ----------- 可以通过下面的式子验证A和b是否真的是PCA矩阵。
    # ynew = npdata @ A.T + b

    # L2 normalization
    row_sums = np.linalg.norm(pca_feat, axis=1)
    pca_feat = pca_feat / row_sums[:, np.newaxis]

    return npdata, pca_feat, A, b

def run_kmeans(x,
               nmb_clusters,
               niter=20,
               verbose=False,
               ngpu=3):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
        list: losses at each iteration
        list: center vector of each cluster
    """
    n_data, d = x.shape

    clus = faiss.Clustering(d, nmb_clusters)
    clus.verbose = True
    clus.niter = niter

    # otherwise the kmeans implementation sub-samples the training set
    clus.max_points_per_centroid = 10000000

    res = [faiss.StandardGpuResources() for i in range(ngpu)]

    flat_config = []
    for i in range(ngpu):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    if ngpu == 1:
        index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
    else:
        indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i])
                   for i in range(ngpu)]
        index = faiss.IndexReplicas()
        for sub_index in indexes:
            index.addIndex(sub_index)
    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    # losses = faiss.vector_to_array(clus.obj)
    stats = clus.iteration_stats
    losses = np.array([stats.at(i).obj for i in range(stats.size())])
    ctrs = faiss.vector_to_array(clus.centroids).reshape(-1, d)
    if verbose:
        for i, l in enumerate(losses):
            print('k-means loss evolution: Iter {}: {}'
                  .format(i, l))

    return [int(n[0]) for n in I], losses, ctrs

class Kmeans(object):
    def __init__(self, k, niter=20):
        self.k = k
        self.niter = niter
        self.ctrs = []
        self.losses = []

    def cluster(self, data, verbose=True):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb, pca_feat, A, b = preprocess_features(data)
        self.xb = xb
        self.pca_feat = pca_feat
        self.pca_A = A
        self.pca_b = b
        print('PCA time: {0:.0f} s'.format(time.time() - end))
        print('PCA shape', pca_feat.shape)

        # cluster the data
        I, losses, ctrs = run_kmeans(pca_feat,
                                     self.k, self.niter,
                                     verbose)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)

        self.losses = losses
        self.ctrs = ctrs
        self.preds = np.array(I)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return losses


# ------------------- load data and merge ---------------------
log_dir = f'/data/space2/huangziyue/DOTA_800_600/train/7_18_Extract_Feats_DOTA_SAM_with_GT_DINOv2_ViTL'
out_dir = f'/data/space2/huangziyue/DOTA_800_600/train/7_26_DOTA_DINOv2_tsne_pca_384'
mkdir(out_dir)
all_feats = []
img_count = 0
feat_count = 0
log_file_list = sorted(list(os.listdir(log_dir)))
feat_ids = []
feat_img_names = []
img_meta_list = []
gt_bboxes = []

for log_file in tqdm(log_file_list):
    # if img_count > 1000:
    #     break
    log_data = pklload(log_dir + '/' + log_file, msg=False)
    feats = log_data['patch_feats']
    img_metas = log_data['img_metas'][0]
    img_stem = Path(img_metas['img_path']).stem
    all_feats.append(feats)
    img_meta_list.append(img_metas)
    img_count += 1
    for f in feats:
        feat_ids.append(feat_count)
        feat_img_names.append(img_stem)
        feat_count += 1

all_feats = np.concatenate(all_feats)

# pklsave(all_feats, merge_feat_pth)

# ------------------- PCA -> Norm -> K-Means ---------------------
feats = all_feats # pklload(merge_feat_pth)
model = Kmeans(k=256, niter=100)
model.cluster(all_feats, verbose=True)
output_info_path = out_dir + '/' + f'KMeans_out.pkl'
output_info = dict(
    preds=model.preds,
    losses=model.losses,
    ctrs=model.ctrs,
    pca_feat=model.pca_feat,
    pca_A=model.pca_A,
    pca_b=model.pca_b,
    log_file_list=log_file_list,
    feat_ids=feat_ids,
    feat_img_names=feat_img_names,
    gt_bboxes=gt_bboxes,
    img_meta_list=img_meta_list
)
pklsave(output_info, output_info_path)
print(f'Feat_num: {len(model.pca_feat)}')
fig_pth = out_dir + f'/K256_tsne_vis.png'
inds = np.arange(len(model.xb))
choise_inds = inds[np.random.choice(inds, min(len(model.xb), 100000))]
tsne_feat = model.xb[choise_inds]
tsne_label = model.preds.flatten()[choise_inds]
tsne2d = TSNE(n_components=2, init='pca', random_state=0)
X_tsne_2d = tsne2d.fit_transform(tsne_feat)
pklsave(X_tsne_2d, out_dir + '/' + f'TSNE_out.pkl')

start_time = time.time()
plot_embedding_2d(X_tsne_2d, tsne_label, fig_pth)
tsne_time = time.time() - start_time
print('tsne_time', tsne_time)
