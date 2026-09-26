"""Shared fixtures/builders for the deltatrace test suite.

All fixtures are tiny, fully synthetic and deterministic — no tangram/commot,
no scanpy clustering, no random sampling — so the tests pin exact numbers.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture
def mapping_and_spatial():
    """A 3-cell x 3-spot mapping with hand-chosen point-mass distributions.

    Spots are placed so the maximum pairwise spot distance is 5 (spot1->spot2):
        spot0 = (0, 0), spot1 = (3, 0), spot2 = (0, 4)

    Cell distributions over spots (rows sum to 1):
        cell0 -> spot0,  cell1 -> spot1,  cell2 -> spot0 (identical to cell0)

    Expected 1-Wasserstein distances (Euclidean ground metric, µm):
        d(cell0, cell1) = |spot0 - spot1| = 3
        d(cell0, cell2) = 0  (identical)
        d(cell1, cell2) = 3
    """
    spots = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    adata_sp = ad.AnnData(
        X=np.zeros((3, 1)),
        obs=pd.DataFrame(index=["s0", "s1", "s2"]),
    )
    adata_sp.obsm["spatial"] = spots

    mapping_X = np.array(
        [
            [1.0, 0.0, 0.0],  # cell0 -> spot0
            [0.0, 1.0, 0.0],  # cell1 -> spot1
            [1.0, 0.0, 0.0],
        ],  # cell2 -> spot0
    )
    M = ad.AnnData(
        X=mapping_X,
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["s0", "s1", "s2"]),
    )
    return M, adata_sp


@pytest.fixture
def sc_with_knn():
    """A 3-cell single-cell object carrying a hand-built expression KNN graph.

    Symmetric edges (cell index pairs), distance / connectivity:
        (0,1): dist 1.0, conn 0.5
        (1,2): dist 2.0, conn 0.8
    """
    n = 3
    rows = [0, 1, 1, 2]
    cols = [1, 0, 2, 1]
    dist_vals = [1.0, 1.0, 2.0, 2.0]
    conn_vals = [0.5, 0.5, 0.8, 0.8]
    distances = sp.csr_matrix((dist_vals, (rows, cols)), shape=(n, n))
    connectivities = sp.csr_matrix((conn_vals, (rows, cols)), shape=(n, n))

    adata = ad.AnnData(
        X=np.zeros((n, 2)),
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
    )
    adata.obsp["distances"] = distances
    adata.obsp["connectivities"] = connectivities
    adata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {"n_neighbors": 2, "method": "umap"},
    }
    return adata


@pytest.fixture
def wd_matrix():
    """A 3x3 symmetric Wasserstein-style distance object aligned to c0,c1,c2.

    d(0,1)=4, d(1,2)=6, d(0,2)=10
    """
    n = 3
    rows = [0, 1, 1, 2, 0, 2]
    cols = [1, 0, 2, 1, 2, 0]
    vals = [4.0, 4.0, 6.0, 6.0, 10.0, 10.0]
    X = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["c0", "c1", "c2"]),
    )


@pytest.fixture
def lineage_inputs():
    """A 4-cell object + a 2-path flow_result with two distinct sink lineages.

    clusters:    c0,c1 -> 'A';  c2 -> 'B';  c3 -> 'C'
    pseudotime:  [0.0, 0.3, 0.7, 1.0]
    path0: c0->c1->c2  weight 2.0  (terminal cluster 'B')
    path1: c0->c1->c3  weight 1.0  (terminal cluster 'C')
    """
    adata = ad.AnnData(
        X=np.zeros((4, 2)),
        obs=pd.DataFrame(
            {
                "clusters": ["A", "A", "B", "C"],
                "pseudotime": [0.0, 0.3, 0.7, 1.0],
            },
            index=["c0", "c1", "c2", "c3"],
        ),
    )
    flow_result = {
        "paths": [
            {"cells": ["c0", "c1", "c2"], "weight": 2.0, "lineage_id": "B"},
            {"cells": ["c0", "c1", "c3"], "weight": 1.0, "lineage_id": "C"},
        ]
    }
    return adata, flow_result
