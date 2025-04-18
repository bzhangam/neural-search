/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.constants;

import org.opensearch.index.mapper.ObjectMapper;
import org.opensearch.index.mapper.RankFeaturesFieldMapper;
import org.opensearch.index.mapper.TextFieldMapper;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import reactor.util.annotation.NonNull;

import java.util.HashMap;
import java.util.Map;

import static org.opensearch.neuralsearch.constants.MappingConstants.PROPERTIES;
import static org.opensearch.neuralsearch.constants.MappingConstants.TYPE;

/**
 * Constants for semantic info
 */
public class SemanticInfoFieldConstants {
    public static final String KNN_VECTOR_DIMENSION_FIELD_NAME = "dimension";
    public static final String KNN_VECTOR_METHOD_FIELD_NAME = "method";
    public static final String KNN_VECTOR_METHOD_ENGINE_FIELD_NAME = "engine";
    public static final String KNN_VECTOR_METHOD_DEFAULT_ENGINE = "faiss";
    public static final String KNN_VECTOR_METHOD_NAME_FIELD_NAME = "name";
    public static final String KNN_VECTOR_METHOD_DEFAULT_NAME = "hnsw";
    public static final String KNN_VECTOR_METHOD_SPACE_TYPE_FIELD_NAME = "space_type";

    public static final String CHUNKS_FIELD_NAME = "chunks";
    public static final String CHUNKS_TEXT_FIELD_NAME = "text";
    public static final String CHUNKS_EMBEDDING_FIELD_NAME = "embedding";

    public static final String MODEL_FIELD_NAME = "model";
    public static final String MODEL_ID_FIELD_NAME = "id";
    public static final String MODEL_NAME_FIELD_NAME = "name";
    public static final String MODEL_TYPE_FIELD_NAME = "type";

    // A parameter to control if we want to make the field searchable.
    public static final String INDEX_FIELD_NAME = "index";

    // Default model info field config will be text, and we will not index it to save some effort since we don't
    // expect users want to search against them.
    private static final Map<String, Object> defaultModelInfoFieldConfig = Map.of(
        TYPE,
        TextFieldMapper.CONTENT_TYPE,
        INDEX_FIELD_NAME,
        Boolean.FALSE
    );
    // Default model info config which will have id, name and type. We will use it as the metadata to help us identify
    // the embedding is generated by what model. This info can be useful when we want to reuse the embedding in the
    // existing doc to help us ensure the existing embedding was generated by the same model as the current one.
    private static final Map<String, Object> defaultModelConfig = Map.of(
        PROPERTIES,
        Map.of(
            MODEL_ID_FIELD_NAME,
            defaultModelInfoFieldConfig,
            MODEL_NAME_FIELD_NAME,
            defaultModelInfoFieldConfig,
            MODEL_TYPE_FIELD_NAME,
            defaultModelInfoFieldConfig
        )
    );

    /**
     * It will construct the index mapping config for the semantic info fields.
     * {
     *     "chunks":{
     *         "type": "nested",
     *         "properties":{
     *             "text":{
     *                 "type": "text"
     *             },
     *             "embedding":{
     *                 ... // embedding config
     *             }
     *         }
     *     },
     *     "model":{
     *         ... // model config
     *     }
     * }
     * @param embeddingConfig Config of the embedding field
     * @return Config of the semantic info fields.
     */
    public static Map<String, Object> getBaseSemanticInfoConfig(@NonNull final Map<String, Object> embeddingConfig) {
        final Map<String, Object> chunksConfig = Map.of(
            TYPE,
            ObjectMapper.NESTED_CONTENT_TYPE,
            PROPERTIES,
            Map.of(CHUNKS_TEXT_FIELD_NAME, Map.of(TYPE, TextFieldMapper.CONTENT_TYPE), CHUNKS_EMBEDDING_FIELD_NAME, embeddingConfig)
        );
        return Map.of(PROPERTIES, Map.of(CHUNKS_FIELD_NAME, chunksConfig, MODEL_FIELD_NAME, defaultModelConfig));
    }

    /**
     * @return basic field config for a text embedding field
     */
    public static Map<String, Object> getBaseTextEmbeddingConfig() {
        final Map<String, Object> config = new HashMap<>();
        config.put(TYPE, KNNVectorFieldMapper.CONTENT_TYPE);
        final Map<String, Object> methodConfig = new HashMap<>();
        methodConfig.put(KNN_VECTOR_METHOD_ENGINE_FIELD_NAME, KNN_VECTOR_METHOD_DEFAULT_ENGINE);
        methodConfig.put(KNN_VECTOR_METHOD_NAME_FIELD_NAME, KNN_VECTOR_METHOD_DEFAULT_NAME);
        config.put(KNN_VECTOR_METHOD_FIELD_NAME, methodConfig);
        return config;
    }

    /**
     * @return basic field config for a sparse embedding field
     */
    public static Map<String, Object> getBaseSparseEmbeddingConfig() {
        final Map<String, Object> config = new HashMap<>();
        config.put(TYPE, RankFeaturesFieldMapper.CONTENT_TYPE);
        return config;
    }
}
