## Import packages=============================================================
import math
import multiprocessing
import random
import resource
import time
import traceback
import warnings
from itertools import repeat

import anndata
import leidenalg as la
import numpy as np
import pandas as pd
from igraph import Graph
from scipy.sparse import coo_matrix, csr_matrix, hstack
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics

from .._prune_features import calc_highly_variable_genes, calc_pca
from ._leiden import run_la_clustering

## End Import packages section=================================================

# flake8: noqa: B902
# flake8: noqa: E266

## Suppress warnings from printing
# warnings.filterwarnings("ignore")


## FUNCTION AND CLASS DOCUMENTATION!!


############################################################################### UTILS
############## Adapted from ......... scedar github
def _parmap_fun(f, q_in, q_out):
    """
    Map function to process.

    Parameters
    ----------
    f : `function`
        Function to run in a given single process.
    q_in :
        Input `multiprocessing.Queue`.
    q_out :
        Output `multiprocessing.Queue`.
    """
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=1):
    """
    Run functions with mutiprocessing.

    parmap_fun() and parmap() are adapted from klaus se's post
    on stackoverflow. https://stackoverflow.com/a/16071616/4638182

    parmap allows map on lambda and class static functions.

    Fall back to serial map when nprocs=1.

    Parameters
    ----------
    f : `function`
        Function to run in parallel single processes.
    X : list of iterables
        List of generators or other iterables containing args for
        specified function.
    nprocs : int
        Number of parallel processes to run

    Returns
    -------
    subsample_size : int
        The number of data points/instances/cells to sample.
    """
    print("in parmap")
    if nprocs < 1:
        raise ValueError(f"nprocs should be >= 1. nprocs: {nprocs}")

    nprocs = min(int(nprocs), multiprocessing.cpu_count())
    # exception handling f
    # simply ignore all exceptions. If exception occurs in parallel queue, the
    # process with exception will get stuck and not be able to process
    # following requests.

    def ehf(x):
        try:
            res = f(x)
        except Exception as e:
            res = e
        return res

    # fall back on serial
    if nprocs == 1:
        return list(map(ehf, X))
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()
    proc = [multiprocessing.Process(target=_parmap_fun, args=(f, q_in, q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()
    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]
    [p.join() for p in proc]
    # maintain the order of X
    ordered_res = [x for i, x in sorted(res)]
    # filename = (
    #    "/project/zunderlab/sarah_data/project_ConsensusClusteringMethod/github_package/v_error_ordered_res_output"
    # )
    # joblib.dump(ordered_res, filename + ".sav")
    for i, x in enumerate(ordered_res):
        if isinstance(x, Exception):
            warnings.warn(f"{x} encountered in parmap {i}th arg {X[i]}")
    return ordered_res


############################################################################### UTILS
def get_subsamp_size(n):  # n==data.shape[0]
    """
    Generate subsample size.

    Calculates subsample size for a single clustering.
    Value is chosen from a gaussian whose center is set
    based on the number of data points/instances/cells
    in the dataset.

    Parameters
    ----------
    n : int
        Number of data points/instances/cells.

    Returns
    -------
    subsample_size : int
        The number of data points/instances/cells to sample.
    """
    oom = math.ceil(n / 1000)
    # print(oom)
    if oom > 1000:  # aka more than 1 mil data points
        mu = 30
    elif oom == 1:  # aka fewer than 1000 data points
        mu = 90
    else:
        oom = 1000 - oom  # so that it scales in the appropriate direction
        mu = ((oom - 1) / (1000 - 1)) * (90 - 30) + 30
    subsample_ratio = random.gauss(mu=mu, sigma=10)
    while subsample_ratio >= 100 or subsample_ratio < 10:
        subsample_ratio = random.gauss(mu=mu, sigma=10)
    ## Calculate subsample size
    subsample_size = int((subsample_ratio / 100) * n)
    return subsample_size


## Get hyperparameters
def get_hyperparameters(k_range, la_res_range, n, metric=None):
    """
    Calculate hyperparameters for a single clustering.

    Parameters
    ----------
    k_range : tuple of (int, int)
        Upper and lower limits for selecting random k for neighborhood
        graph construction.
    la_res_range : tuple of (int, int)
        Upper and lower limits for selecting random resolution
        parameter for leiden community detection.
    n : int
        Number of data points/instances/cells.
    metric : {‘euclidean’, ‘cosine’, None}, optional
        Which distance metric to use for calculating kNN graph for clustering.
        For now, one of ('cosine', 'euclidean'), plan to add
        correlation when I can find a fast enough implementation.

    Returns
    -------
    k : int
        Number of neighbors for neighborhood graph construction.
    la_res : int
        Resolution parameter for leiden community detection.
    metric : str
        Which distance metric to use for calculating kNN graph for clustering.
        For now, one of ('cosine', 'euclidean'), plan to add
        correlation when I can find a fast enough implementation.
    subsample_size : int
        The number of data points/instances/cells to sample.
    """
    k = random.sample(range(k_range[0], k_range[1]), 1)[0]
    la_res = random.sample(range(la_res_range[0], la_res_range[1]), 1)[0]
    if metric is None:
        metric = ["euclidean", "cosine"][random.sample(range(2), 1)[0]]
    subsample_size = get_subsamp_size(n)
    return k, la_res, metric, subsample_size


def run_pca_dim_reduction(X):
    """
    Produce PCA-reduced data matrix.

    Generates a dimensionality-reduced data matrix through
    PCA feature extraction. Other methods of feature extraction
    and selection will be included in future releases.

    Parameters
    ----------
    X : :class:`~numpy.array` or :class:`~scipy.sparse.spmatrix`
        Data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.

    Returns
    -------
    X_pca : :class:`~numpy.array` or :class:`~scipy.sparse.spmatrix`
        Data matrix of shape `n_obs` × `n_pcs`. Rows correspond
        to cells and columns to PCA-extracted features.
    """
    time.time()
    if X.shape[1] > 6000:
        bool_features = calc_highly_variable_genes(X)
        X = X[:, bool_features]
    X_pca = np.array(calc_pca(X))
    return X_pca


def run_base_clustering(args_in):
    """
    Run a single iteration of leiden clustering.

    Parameters
    ----------
    args_in : zip
        List containing each hyperparameter required for one round of
        clustering (k, la_res, metric, subsample_size) as well as a
        copy of the data as numpy.array or scipy.sparse.spmatrix

    Returns
    -------
    A list of the following:
    [coo_matrix(c).tocsr(), np.expand_dims(per_iter_clust_assigns, axis=1)]

    coo_matrix(c).tocsr() : :class:`~scipy.sparse.spmatrix`
        Matrix of dimensions n (total number of data points) by
        m (number of clusters) and filled with 1/0 binary occupancy
        of data point per cluster.
    np.expand_dims(per_iter_clust_assigns, axis=1) : ndarray of shape (n, 1)
        One-dimensional array containing zeroes for data points that were not
        sampled, cluster id +1 for sampled data points.
    """
    try:
        data = args_in[1]

        hyperparams = args_in[0]
        iter_k = hyperparams[0]
        la_res = hyperparams[1]
        metric = hyperparams[2]
        subsample_size = hyperparams[3]
        # print('3.0 Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        ## Get indices for random subsample
        subsample_ids = random.sample(range(data.shape[0]), subsample_size)
        ## Subsample data
        n_orig = data.shape[0]  # save original number of data points
        print(n_orig)
        data = data[subsample_ids, :]
        print(n_orig)
        print("3.1 Memory usage: %s (kb)" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        ## Log transform features if it is scRNAseq that has not yet been transformed
        if (
            data.shape[1] > 8000
        ):  # internal heuristic for if it scrna seq, def needs to change (just require data already be log transformed)
            # log2 transform, if it is not already! (can check this my looking at max value in array)
            if np.max(data) > 20:
                data = np.log1p(data)
                print("log transformed, max=" + str(np.max(data)))

        ### Approximate test for whether data needs to be scaled
        if np.std(np.max(data, axis=0)) > 5:
            raise Exception(
                "Dataset must be scaled in a manner appropriate for your data type before running through SHaRC"
            )

        ## Dimensionality reduction if large number of features
        data = run_pca_dim_reduction(data)
        print("3.2 Memory usage: %s (kb)" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        ## Run leiden clustering
        clusters = run_la_clustering(X=data, k=iter_k, la_res=la_res / 100, metric=metric)
        print("Clusters type: " + str(type(clusters)))
        print("Clusters shape: " + str(clusters.shape))
        print(clusters[0, 0:10])
        print("3.3 Memory usage: %s (kb)" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        ## Prepare outputs for this ensemble member
        per_iter_clust_assigns = np.zeros((n_orig), dtype=np.uint8)
        print(per_iter_clust_assigns[0:10])
        per_iter_clust_assigns[subsample_ids] = clusters + 1
        print(per_iter_clust_assigns[0:10])
        # print(per_iter_clust_assigns[0:50])

        n_clust = len(np.unique(clusters))
        print(n_clust)
        a = np.zeros((n_orig), dtype=np.uint8)
        a[subsample_ids] = clusters[0] + 1
        print(a[0:10])
        b = np.ones((n_orig), dtype=np.uint8)
        c = np.zeros((n_orig, len(np.unique(a))), dtype=np.uint8)
        # print(c[0:50,0:3])
        np.put_along_axis(arr=c, indices=np.expand_dims(a, axis=1), values=np.expand_dims(b, axis=1), axis=1)  # )#,
        # print(c[0:50,0:3])
        c = np.delete(c, 0, 1)
        # print(c[0:50,0:3])
        print("3.4 Memory usage: %s (kb)" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception as ex:
        traceback.print_exception(type(ex), ex, ex.__traceback__)
        return ["error", data]
    return [coo_matrix(c).tocsr(), np.expand_dims(per_iter_clust_assigns, axis=1)]


def get_hard_soft_clusters(n, clustering, bg):
    """
    Generate hard and soft clusters for a single bipartite clustering.

    Parameters
    ----------
    n : int
        Number of data points/instances/cells.
    clustering : ndarray of shape (n_cons_clust, 1)
        Consensus cluster assignment for each ensemble cluster.
    bg :  :class:`~igraph.Graph`
        The bipartite graph generated from the ensemble of clusterings.

    Returns
    -------
    hard_clusters : int
        Hard cluster assignments for every sample.
    soft_membership_matrix : :class:`numpy.ndarray`
        Contains mebership values for each sample in each consensus cluster.
    """
    clusters_vertex_ids = np.array(bg.vs.indices)[[x >= n for x in bg.vs.indices]]
    cells_clusts = np.unique(clustering)
    clust_occ_arr = np.zeros((n, len(cells_clusts)), int)
    for v in range(len(cells_clusts)):
        cluster_id = cells_clusts[v]
        cluster_memb = [
            clusters_vertex_ids[i] for i, j in enumerate(clustering) if j == cluster_id
        ]  # np.array(vertex_names)[]
        node_subset, counts = np.unique(
            [e.source for e in bg.es.select(_source_in=cluster_memb)], return_counts=True
        )  # +self.n_cells
        clust_occ_arr[node_subset, v] = counts  # [restricted_hg.degree(node) for node in restricted_hg.nodes]
    # hard_clusters = np.argmax(clust_occ_arr, axis=1)
    # ^^ is biased, use the following to randomly break ties:
    hard_clusters = np.array([np.random.choice(np.where(row == row.max())[0]) for row in clust_occ_arr])
    # clust_occ_arr = clust_occ_arr[:,np.unique(hard_clusters)]
    soft_membership_matrix = clust_occ_arr / clust_occ_arr.sum(axis=1, keepdims=True)  # [:,hard_clusters]
    # soft_membership_matrix = soft_membership_matrix[:,np.unique(hard_clusters)]
    # soft_membership_matrix = np.divide(soft_membership_matrix,soft_membership_matrix.sum(axis=1))
    # hard_clusters = pd.Categorical(np.argmax(soft_membership_matrix, axis=1))
    return hard_clusters, soft_membership_matrix  # , clust_occ_arr


def consensus_cluster_leiden(in_args):
    """
    Runs a single iteration of leiden clustering.

    Parameters
    ----------
    args_in : zip
        List containing (1) the number of data points, (2) the bipartite
        leiden clustering resolution, and (3) the bipartite graph generated
        from the ensemble of clusterings.

    Returns
    -------
    hard_clusters :  :class:`~pandas.Series`
        Categorical series containing hard cluster assignments per data point.
    csr_matrix(soft_membership_matrix) : :class:`~scipy.sparse.spmatrix`
        Matrix of dimensions n (total number of data points) by
        m (number of consensus clusters) and filled with membership ratio
        of data point per cluster.
    ari : float
        Adjusted Rand Index between cluster membership assignments derived
        from clustering of clusters and majority voting for hard assignments
        versus the hard cluster memberships of the data points directly
        resulting from the bipartite clustering.
    i : float
        Bipartite leiden resolution parameter, a sanity check to ensure
        parallel processing maintains expected order of resolutions
        in output.
    """
    ## Run initial Lieden clustering with specified resolution value
    n = in_args[0]
    i = in_args[1]
    # print(i)
    bg = in_args[2]  # [0]
    print("6 Memory usage: %s (kb)" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # print(objgraph.show_most_common_types())
    p_01, p_0, p_1 = la.CPMVertexPartition.Bipartite(bg, resolution_parameter_01=i)
    optimiser = la.Optimiser()
    diff = optimiser.optimise_partition_multiplex(partitions=[p_01, p_0, p_1], layer_weights=[1, -1, -1])
    print("6.1 Memory usage: %s (kb)" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    clustering = np.array(p_01.membership)[np.where(bg.vs["type"])[0]]  # just select clusters assigns for clusters
    print("6.2 Memory usage: %s (kb)" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    clustering_cells = np.array(p_01.membership)[
        [i for i, val in enumerate(bg.vs["type"]) if not val]
    ]  # just select clusters assigns for cells?
    hard_clusters, soft_membership_matrix = get_hard_soft_clusters(n, clustering, bg)  # , clust_occ_arr
    # return clust_occ_arr
    print("6.3 Memory usage: %s (kb)" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    ## Get ids of the base clusters that are participating in hard clusters (== good, keep)
    ## This trims "outlier" clusters that arose during the ensemble clustering
    hard_only = list(set(range(soft_membership_matrix.shape[1])).intersection(set(np.unique(hard_clusters))))
    soft_only_cluster_memb = [i + n for i, j in enumerate(clustering) if j not in hard_only]
    ## Create new bipartite graph with outlier base clusters removed
    bg.delete_vertices(soft_only_cluster_memb)
    ## Run clustering again with remaining base clusters
    p_01, p_0, p_1 = la.CPMVertexPartition.Bipartite(bg, resolution_parameter_01=i)
    optimiser = la.Optimiser()
    diff = optimiser.optimise_partition_multiplex(partitions=[p_01, p_0, p_1], layer_weights=[1, -1, -1])
    clustering = np.array(p_01.membership)[np.where(bg.vs["type"])[0]]  # just select clusters assigns for clusters
    hard_clusters, soft_membership_matrix = get_hard_soft_clusters(n, clustering, bg)
    # keep only clusters that are majority membership for at least one data point
    soft_membership_matrix = soft_membership_matrix[:, np.unique(hard_clusters)]
    print("6.41 Memory usage: %s (kb)" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # convert resulting membership back to ratio*
    soft_membership_matrix = np.divide(soft_membership_matrix, soft_membership_matrix.sum(axis=1)[:, None])  # [:,None]
    print("6.42 Memory usage: %s (kb)" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # calculate final hard clusters based on
    hard_clusters = pd.Categorical(
        np.array([np.random.choice(np.where(row == row.max())[0]) for row in soft_membership_matrix])
    )
    print("6.43 Memory usage: %s (kb)" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # print("row sums: " + str(np.unique(final_smm.sum(axis=1))))

    ## Get ari/ami between cells clusters and cluster clusters
    metrics.adjusted_mutual_info_score(hard_clusters, clustering_cells)
    ari = metrics.adjusted_rand_score(hard_clusters, clustering_cells)
    print("7 Memory usage: %s (kb)" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # print(objgraph.show_most_common_types())

    return hard_clusters, csr_matrix(soft_membership_matrix), ari, i


############################################################################### CLASS


class ConsensusCluster:
    """
    Consensus Clustering.

    Runs ensemble of leiden clusterings on random subsamples of input data with
    random hyperparameters from within the default range (or user specified).
    Then generates a bipartite graph from these results where data instances
    have edges to all clusters they were assigned to accross the ensemble of
    clsuterings. Bipartite community detection is run on this resulting graph
    to obtain final hard and soft clusters.

    Parameters
    ----------
    reduction : {'all', ‘pca’}
        Which method to use for featureextraction/selection/dimensionality
        reduction, or `all` for use all features. Currently only PCA is
        supported, but alternative options will be added in future releases.
        Once other options are added, the default will be to randomly select
        a reduction for each ensemble member. For datasets with fewer than
        10 features, all features are used.
    metric : {'euclidean', 'cosine', None}
        Metric used for neighborhood graph construction. Can be "euclidean",
        "cosine", or `None`. Default is `None`, in which case the metric is
        randomly selected for each ensemble member. Other metrics will be
        added in future releases.
    ensemble_size : int, default=150
        Number of clusterings to run in the ensemble.
    k_range : tuple of (int, int)
        Upper and lower limits for selecting random k for neighborhood
        graph construction.
    la_res_range : tuple of (int, int)
        Upper and lower limits for selecting random resolution
        parameter for leiden community detection.
    nprocs : int, default=None
        How many processes to run in parallel. If `None`, value is set using
        `multiprocessing.cpu_count()` to find number of available cores.
        This is used as a check and the minimum value between number of
        cores detected and specified number of processes is set as final value.

    Attributes
    ----------
    per_iter_clust_assigns : :class:`~scipy.sparse.csr_matrix`
        Each column is an iteration of the ensemble of clusterings, rows are
        data points, and array is filled with cluster assignments per iteration
        from 1 to m clusters. 0 indicates the data point was not included in
        that ensemble member.
    bipartite : :class:`~igraph.Graph`
        The bipartite graph generated from the ensemble of clusterings, where
        data points have connections to the cluster they were assigned to in
        each ensemble member where they were included.
    multiresolution_clusters : DataFrame
        Cluster assignments resulting from all resolutions tested for
        final consensus, concatenated into one DataFrame. Each column
        is a difference resolution.
    hard_clusters : array-like
        Final hard cluster assignments.
    soft_membership_metrix : array-like
        Matrix with each sample's membership in each final consensus cluster.
    cell_conf_score : array-like
        Confidence scores per sample.
    adata
        Annotated data matrix. Hard cluster assignments and soft cluster
        memberships are stored as annotations. See `AnnData Documentation
        <https://anndata.readthedocs.io/en/latest/index.html>`_ for more info.
    """

    __slots__ = (
        "reduction",
        "metric",
        "ensemble_size",
        "auto_stop",
        "k_range",
        "la_res_range",
        "nprocs",
        "per_iter_clust_assigns",
        "bipartite",
        "multiresolution_clusters",
        "hard_clusters",
        "soft_membership_matrix",
        "cell_conf_score",
        "adata",
    )

    def __init__(
        self,
        reduction="pca",
        metric=None,  # how to add options?
        ensemble_size=150,
        auto_stop=False,
        k_range=(25, 175),
        la_res_range=(70, 170),
        nprocs=None,
    ):
        self.reduction = reduction
        self.metric = metric
        self.ensemble_size = ensemble_size
        self.auto_stop = auto_stop
        self.k_range = (int(k_range[0]), int(k_range[1]))
        self.la_res_range = (int(la_res_range[0]), int(la_res_range[1]))
        if nprocs is None:
            nprocs = multiprocessing.cpu_count()
        self.nprocs = min(int(nprocs), multiprocessing.cpu_count())

    def ensemble(self, data):
        """
        Run ensemble of clusterings.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The data.

        Returns
        -------
        bipartite : :class:`~igraph.Graph`
            The bipartite graph generated from the ensemble of clusterings, where
            data points have connections to the cluster they were assigned to in
            each ensemble member where they were included.
        per_iter_clust_assigns : :class:`~scipy.sparse.csr_matrix`
            Each column is an iteration of the ensemble of clusterings, rows are
            data points, and array is filled with cluster assignments per iteration
            from 1 to m clusters. 0 indicates the data point was not included in
            that ensemble member.
        """
        start_time = time.time()

        if not isinstance(data, csr_matrix):
            sparsity = 1.0 - np.count_nonzero(data) / data.size
            print("Sparsity: " + str(sparsity))
            if sparsity > 0.1:
                data = csr_matrix(data)

        data_iterator = repeat(data, self.ensemble_size)
        # to handle (rare) cases where number of data points is less that the
        # max limit on range of values for k (# neighbors)
        if data.shape[0] < self.k_range[1]:
            self.k_range = (self.k_range[0], data.shape[0] - 1)
        print("iter_k: " + str(self.k_range))
        print("la_res: " + str(self.la_res_range))
        hyperparam_iterator = [
            get_hyperparameters(self.k_range, self.la_res_range, data.shape[0], self.metric)
            for x in range(self.ensemble_size)
        ]
        args = list(zip(hyperparam_iterator, data_iterator))

        print("starting ensemble clustering multiprocess")
        out = np.array(parmap(run_base_clustering, args, nprocs=self.nprocs))

        # idxs = np.where(out[:, 0] == "error")[0]
        # if idxs.shape[0] > 0:
        #    for idx in idxs:
        #        if isinstance(out[idx, 1], csr_matrix):
        #            pd.DataFrame(out[idx, 1].toarray()).to_csv(
        #                os.path.join(out_dir, str(idx) + "error_data_out.csv"), index=None
        #            )
        #        else:
        #            pd.DataFrame(out[idx, 1]).to_csv(os.path.join(out_dir, str(idx) + "error_data_out.csv"), index=None)
        try:
            clust_out = hstack(out[:, 0])
            # filename="/project/zunderlab/sarah_data/project_ConsensusClusteringMethod/github_package/v_no_error_output"
            # joblib.dump(out, out_dir + "v_no_error_output.sav")
        except Exception:
            print("consensus_cluster.py, line 444, in fit: clust_out = hstack(out[:,0])")
            print("Error: ")
            # try:
            #    print(e)
            # except:
            #    print("e wont print")
            # filename="/project/zunderlab/sarah_data/project_ConsensusClusteringMethod/github_package/v_" + self.reduction + "_" + self.metric + "_" + str(self.iter_k_range) + "_" + str(self.la_res_range) + "_" + "error_output"
            # joblib.dump(out, out_dir + "/ensemble_out_error.sav")
            # with open(filename + ".txt", "w") as f:
            #    f.writelines(f"{place for place in out}\n")
            # with open(filename + '.data', 'wb') as filehandle:
            #    # Store the data as a binary data stream
            #    pickle.dump(out, filehandle)

        per_iter_clust_assigns = csr_matrix(coo_matrix(np.concatenate(out[:, 1], axis=1)))

        bipartite = Graph(
            np.concatenate(
                (np.expand_dims(clust_out.row, axis=1), np.expand_dims(clust_out.col + data.shape[0], axis=1)), axis=1
            )
        ).as_undirected()
        type_ls = [0] * data.shape[0]  # self.
        type_ls.extend([1] * (bipartite.vcount() - data.shape[0]))  # self.
        bipartite.vs["type"] = type_ls  # self.
        assert bipartite.is_bipartite()

        finish_time = time.perf_counter()
        print(f"Ensemble clustering finished in {finish_time-start_time} seconds")

        return bipartite, per_iter_clust_assigns

    def consensus(self, n, bg, out_dir=None):
        """
        Find consensus from ensemble of clusterings.

        Parameters
        ----------
        n : int
            Number of data points/instances/cells.
        bg :  :class:`~igraph.Graph`
            The bipartite graph generated from the ensemble of clusterings.

        Returns
        -------
        hard_clusters : :class:`numpy.ndarray`
            Final hard cluster assignments for every sample.
        soft_membership_matrix : :class:`numpy.ndarray`
            Contains mebership values for each sample in each final consensus
            cluster.
        """
        ## Run final consensus
        start_time2 = time.time()

        res_ls = [
            0.05,
            0.075,
            0.1,
            0.125,
            0.15,
            0.175,
            0.2,
            0.225,
            0.25,
            0.275,
            0.3,
            0.325,
            0.35,
            0.375,
            0.4,
            0.425,
            0.45,
            0.475,
            0.5,
            0.525,
            0.55,
            0.575,
            0.6,
            0.625,
            0.65,
            0.675,
            0.7,
        ]

        print("starting consensus multiprocess")
        start_time = time.perf_counter()
        bg_iterator = repeat(bg, len(res_ls))
        n_iterator = repeat(n, len(res_ls))
        args = list(zip(n_iterator, res_ls, bg_iterator))
        out = np.array(parmap(consensus_cluster_leiden, args, nprocs=self.nprocs))
        try:
            out = out[np.argsort(out[:, 3])]
            # filename="/project/zunderlab/sarah_data/project_ConsensusClusteringMethod/github_package/v_no_error_output"
            # joblib.dump(out, out_dir + "v_no_error_output.sav")
        except Exception as e:
            print("consensus_cluster.py, line ~665, in transform: out = out[np.argsort(out[:,3])]")
            print("Error: ")
            print(e)
            # try:
            #    print(e)
            # except:
            #    print("e wont print")
            # joblib.dump(out, out_dir + "/consensus_out_error.sav")
        # assert out[:, 3] == res_ls, "Warning: resolution order is wrong"

        indices = [
            index for index, value in sorted(enumerate(list(out[:, 3])), key=lambda x: x[1])
        ]  # in case parallel returned in wrong order
        all_clusterings = [pd.DataFrame(x, dtype=int) for x in out[indices, 0]]
        all_clusterings_df = pd.concat(all_clusterings, axis=1)
        all_clusterings_df.columns = list(range(all_clusterings_df.shape[1]))
        self.multiresolution_clusters = all_clusterings_df
        # print(objgraph.show_most_common_types())
        # "out" contains an array of: clustering, hard_clusters, soft_membership_matrix, bg2, ari, ami, res
        finish_time = time.perf_counter()
        print(f"Program finished in {finish_time-start_time} seconds")

        # find res with minimum sum of distances to all other res
        dist = pdist(
            self.multiresolution_clusters.T, metric=lambda u, v: 1 - metrics.adjusted_rand_score(u, v)
        )  # metric='correlation' lambda u, v: metrics.adjusted_rand_score(u,v))
        opt_res_idx = np.argmin(squareform(dist).sum(axis=0))
        # extract final hard and soft clusters for selected optimal resolution
        hard_clusters = out[opt_res_idx, 0]  # kl.knee
        soft_membership_matrix = out[opt_res_idx, 1].toarray()
        # print("Final res: " + str(res_ls[np.argmax(acc_ls)]))
        print("Final res: " + str(res_ls[opt_res_idx]))

        time_per_iter = time.time() - start_time2
        print("time to run final consensus: " + str(time_per_iter))
        return hard_clusters, soft_membership_matrix

    def consensus_cluster(self, data, out_dir=None):
        """
        Run ensemble of clusterings and find consensus.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The data.

        Returns
        -------
        `ConsensusCluster` object modified in place.
        """
        start_time = time.time()
        bipartite, per_iter_clust_assigns = self.ensemble(data)
        ## Add info to object for post process access
        self.per_iter_clust_assigns = per_iter_clust_assigns
        self.bipartite = bipartite

        hard_clusters, soft_membership_matrix = self.consensus(n=data.shape[0], bg=bipartite, out_dir=out_dir)

        print("Final Clustering:")
        print("n hard clusters: " + str(len(np.unique(hard_clusters))))
        print("n soft clusters: " + str(soft_membership_matrix.shape[1]))
        self.hard_clusters = hard_clusters
        self.soft_membership_matrix = soft_membership_matrix
        self.cell_conf_score = np.max(soft_membership_matrix, axis=1)
        time_per_iter = time.time() - start_time
        print("Full runtime: " + str(time_per_iter))

    def make_adata(self, data, feature_names=None, sample_names=None, return_adata=False):
        """
        Make annotated data object.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The data.
        feature_names : array-like of shape (n_features), default = None
            Feature names of the data.
        sample_names : array-like of shape (n_samples), default = None
            Sample names of the data.
        return_adata : bool, default `True`
            Whether to return the adata object (if `True`) or only add as an
            attribute to the ConsensusCluster object (if `False`,
            which is default).

        Returns
        -------
        `ConsensusCluster` object modified in place or `anndata.AnnData`.
        """
        self.adata = anndata.AnnData(X=data)

        if feature_names is not None:
            self.adata.var_names = feature_names
        if sample_names is not None:
            self.adata.obs = pd.DataFrame({"sample_names": sample_names})

        self.adata.obs["hard_clusters"] = self.hard_clusters
        self.adata.obsm["soft_membership_matrix"] = self.soft_membership_matrix
        self.adata.obs["cell_conf_score"] = self.cell_conf_score

        if not return_adata:
            return self.adata