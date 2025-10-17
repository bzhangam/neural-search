/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.neuralsearch.jni;

public class VsagSparseDataset {
    // Number of elements in the dataset
    private int numElements;

    // 2D array: ids for each vector [vector][dim]
    private int[][] ids;

    // 2D array: values for each vector [vector][dim]
    private float[][] values;

    // Optional: global IDs for vectors (used for recall computation, etc.)
    private long[] globalIds;

    public VsagSparseDataset(int numElements, int[][] ids, float[][] values, long[] globalIds) {
        this.numElements = numElements;
        this.ids = ids;
        this.values = values;
        this.globalIds = globalIds;
    }

    public int getNumElements() { return numElements; }
    public int[][] getIds() { return ids; }
    public float[][] getValues() { return values; }
    public long[] getGlobalIds() { return globalIds; }
}
