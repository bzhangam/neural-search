/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.jni;

public class VsagSearchResult {
    // The ID of the matched vector
    private long id;

    // The similarity (score) value
    private float score;

    public VsagSearchResult(long id, float score) {
        this.id = id;
        this.score = score;
    }

    public long getId() {
        return id;
    }

    public float getScore() {
        return score;
    }
}
