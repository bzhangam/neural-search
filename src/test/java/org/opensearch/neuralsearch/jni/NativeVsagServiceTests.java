/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.jni;

import org.opensearch.test.OpenSearchTestCase;

import java.util.Arrays;

public class NativeVsagServiceTests extends OpenSearchTestCase {

    public void test_add() {
        NativeVsagService service = new NativeVsagService();
        service.init();

        /*
         * build_params is the configuration for building a sparse index.
         *
         * - dtype: Must be set to "sparse", indicating the data type of the vectors.
         * - dim: Dimensionality of the sparse vectors (must be >0, but does not affect the result).
         * - metric_type: Distance metric type, currently only "ip" (inner product) is supported.
         * - index_param: Parameters specific to sparse indexing:
         *   - use_reorder: If true, enables full-precision re-ranking of results. This requires storing additional data.
         *     When doc_prune_ratio is 0, use_reorder can be false while still maintaining full-precision results.
         *   - term_id_limit: Maximum term id (e.g., when term_id_limit = 10, then, term [15: 0.1] in sparse vector is not allowed)
         *   - doc_prune_ratio: Ratio of term pruning in documents (0 = no pruning).
         *   - window_size: Window size for table scanning. Related to L3 cache size; 100000 is an empirically optimal value.
         */
        String buildParams = """
        {
            "dtype": "sparse",
            "dim": 128,
            "metric_type": "ip",
            "index_param": {
                "use_reorder": true,
                "term_id_limit": 1000000,
                "doc_prune_ratio": 0.0,
                "window_size": 100000
            }
        }
        """;

        long indexPointer = service.createIndex("sindi", buildParams);
        System.out.println("indexPointer: " + indexPointer);

        VsagSparseVector[] sparseVectors = new VsagSparseVector[10];
        for (int i = 0; i < 10; i++) {
            VsagSparseVector sparseVector = new VsagSparseVector(1, new int[]{1}, new float[]{(float) i /2}, i);
            sparseVectors[i] = sparseVector;
        }

        VsagDataset dataset = new VsagDataset(sparseVectors);
        long[] failedIds = service.add(indexPointer, dataset);
        System.out.println("failedIds: " + Arrays.toString(failedIds));

        /*
         * search_params is the configuration for sparse index search.
         *
         * - sindi: Parameters specific to sparse indexing search:
         *   - query_prune_ratio: Ratio of term pruning for the query (0 = no pruning).
         *   - n_candidate: Number of candidates for re-ranking. Must be greater than topK.
         *     This parameter is ignored if use_reorder is false in the build parameters.
         */
        String search_params = """
        {
            "sindi": {
                "query_prune_ratio": 0,
                "n_candidate": 10
            }
        }
        """;
        VsagSparseVector sparseVector = new VsagSparseVector(1, new int[]{1}, new float[]{0.5f}, 0);
        VsagDataset queryDataset = new VsagDataset(new VsagSparseVector[]{sparseVector});
        VsagSearchResult[] results = service.knnSearch(indexPointer, queryDataset, 10, search_params);
        System.out.println("results: " + results.length);

        String filePath = "/app/index/sindi_index";
        service.serializeIndex(indexPointer, filePath);

        long newIndexPointer = service.createIndex("sindi", buildParams);

        service.deserializeIndex(newIndexPointer, filePath);
        VsagSearchResult[] newResults = service.knnSearch(newIndexPointer, queryDataset, 10, search_params);
        System.out.println("new results: " + newResults.length);

        service.cleanup(indexPointer);
    }
}
