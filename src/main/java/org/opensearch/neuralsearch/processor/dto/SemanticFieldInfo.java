/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.processor.dto;

import lombok.Data;

import java.util.List;

import static org.opensearch.neuralsearch.constants.MappingConstants.PATH_SEPARATOR;
import static org.opensearch.neuralsearch.constants.SemanticInfoFieldConstants.CHUNKS_EMBEDDING_FIELD_NAME;
import static org.opensearch.neuralsearch.constants.SemanticInfoFieldConstants.CHUNKS_FIELD_NAME;
import static org.opensearch.neuralsearch.constants.SemanticInfoFieldConstants.MODEL_FIELD_NAME;

@Data
public class SemanticFieldInfo {
    private String value;
    private String modelId;
    private String fullPath;
    private String semanticInfoFullPath;
    private List<String> chunks;

    public String getFullPathForChunks() {
        return semanticInfoFullPath + PATH_SEPARATOR + CHUNKS_FIELD_NAME;
    }

    public String getFullPathForEmbedding(int index) {
        return semanticInfoFullPath + PATH_SEPARATOR + CHUNKS_FIELD_NAME + PATH_SEPARATOR + index + PATH_SEPARATOR
            + CHUNKS_EMBEDDING_FIELD_NAME;
    }

    public String getFullPathForModelInfo() {
        return semanticInfoFullPath + PATH_SEPARATOR + MODEL_FIELD_NAME;
    }
}
