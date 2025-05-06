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

/**
 * SemanticFieldInfo is a data transfer object to help hold semantic field info
 */
@Data
public class SemanticFieldInfo {
    /**
     * The raw string value of the semantic field
     */
    private String value;
    /**
     * The model id of the semantic field which will be used to generate the embedding
     */
    private String modelId;
    /**
     * The full path to the semantic field
     */
    private String fullPath;
    /**
     * The full path to the semantic info fields
     */
    private String semanticInfoFullPath;
    /**
     * The chunked strings of the original string value of the semantic field
     */
    private List<String> chunks;

    /**
     * @return full path to the chunks field of the semantic field
     */
    public String getFullPathForChunks() {
        return semanticInfoFullPath + PATH_SEPARATOR + CHUNKS_FIELD_NAME;
    }

    /**
     * @param index index of the chunk the embedding is in
     * @return full path to the embedding field of the semantic field
     */
    public String getFullPathForEmbedding(int index) {
        return semanticInfoFullPath + PATH_SEPARATOR + CHUNKS_FIELD_NAME + PATH_SEPARATOR + index + PATH_SEPARATOR
            + CHUNKS_EMBEDDING_FIELD_NAME;
    }

    /**
     * @return full path to the model info fields
     */
    public String getFullPathForModelInfo() {
        return semanticInfoFullPath + PATH_SEPARATOR + MODEL_FIELD_NAME;
    }
}
