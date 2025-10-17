/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.jni;

/**
 * Wrap the c++ implementation of the Sparse Inverted Non-redundant Distance Index
 */
public class NativeVsagService {
    static {
        System.loadLibrary("opensearch_neural_vsag"); //
    }

    // Native method to initialize VSAG library (vsag::init())
    public native void init();

    /**
     *
     * @param indexType
     * @param buildParams
     * @return indexPointer
     */
    public native long createIndex(String indexType, String buildParams);

    // Native method to build the index (index->Build(base))
    public native boolean add(long indexPointer, VsagSparseDataset baseDataset);

    // Native method to serialize index to disk
    public native boolean serializeIndex(long indexPointer, String filePath);

    // Native method to deserialize index from disk
    public native boolean deserializeIndex(long indexPointer, String filePath, String buildParams);

    // Native method to perform a KNN search
    public native VsagSearchResult[] knnSearch(
            long indexPointer, VsagSparseDataset queryDataset, int k, String searchParams
    );

    // Native method to clean up resources, if needed
    public native void cleanup(long indexPointer);
}
