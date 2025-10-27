/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.jni;

public class VsagDataset {
    private VsagSparseVector[] sparseVectors;

    public VsagDataset(VsagSparseVector[] sparseVectors) {
        this.sparseVectors = sparseVectors;
    }

    public VsagSparseVector[] getSparseVectors() {
        return sparseVectors;
    }
}
