/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.neuralsearch.jni;

public class VsagSparseVector {
    private int length;
    private int[] ids;
    private float[] values;
    private long docId;

    public VsagSparseVector(int length, int[] ids, float[] values, long docId) {
        this.length = length;
        this.ids = ids;
        this.values = values;
        this.docId = docId;
    }

    public int getLength() {
        return length;
    }

    public int[] getIds() {
        return ids;
    }

    public float[] getValues() {
        return values;
    }

    public long getDocId() {
        return docId;
    }
}
