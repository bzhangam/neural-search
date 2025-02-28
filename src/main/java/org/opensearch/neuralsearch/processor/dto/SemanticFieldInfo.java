/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.processor.dto;

import lombok.Data;
import org.apache.commons.lang3.tuple.Pair;

import java.util.List;

@Data
public class SemanticFieldInfo {
    private String value;
    private String modelId;
    private String fullPath;
    private String semanticInfoFullPath;
    private List<String> chunks;
    private List<Pair<String, List<Float>>> chunkToEmbeddingMapFromExistingDoc;
    private List<Pair<String, List<Float>>> chunkToEmbeddingMapFromRequest;

    public String getFullPathForChunkedText(int index) {
        return semanticInfoFullPath + ".chunks." + index + ".text";
    }

    public String getFullPathForChunks() {
        return semanticInfoFullPath + ".chunks";
    }

    public String getFullPathForEmbedding(int index) {
        return semanticInfoFullPath + ".chunks." + index + ".embedding";
    }

    public String getFullPathForModelInfo(int index) {
        return semanticInfoFullPath + ".model";
    }
}
