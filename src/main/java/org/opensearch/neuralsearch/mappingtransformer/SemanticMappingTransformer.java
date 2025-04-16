/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.mappingtransformer;

import com.google.common.annotations.VisibleForTesting;
import joptsimple.internal.Strings;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.index.mapper.MappingTransformer;

import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.model.TextEmbeddingModelConfig;
import org.opensearch.neuralsearch.constants.MappingConstants;
import org.opensearch.neuralsearch.constants.SemanticFieldConstants;
import org.opensearch.neuralsearch.ml.MLCommonsClientAccessor;
import reactor.util.annotation.NonNull;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import static org.opensearch.neuralsearch.constants.SemanticFieldConstants.MODEL_ID;
import static org.opensearch.neuralsearch.constants.SemanticFieldConstants.SEMANTIC_INFO_FIELD_NAME;
import static org.opensearch.neuralsearch.constants.SemanticInfoFieldConstants.KNN_VECTOR_DIMENSION_FIELD_NAME;
import static org.opensearch.neuralsearch.constants.SemanticInfoFieldConstants.KNN_VECTOR_METHOD_FIELD_NAME;
import static org.opensearch.neuralsearch.constants.SemanticInfoFieldConstants.KNN_VECTOR_METHOD_SPACE_TYPE_FIELD_NAME;
import static org.opensearch.neuralsearch.constants.SemanticInfoFieldConstants.getBaseSemanticInfoConfig;
import static org.opensearch.neuralsearch.constants.SemanticInfoFieldConstants.getBaseSparseEmbeddingConfig;
import static org.opensearch.neuralsearch.constants.SemanticInfoFieldConstants.getBaseTextEmbeddingConfig;
import static org.opensearch.neuralsearch.mappingtransformer.SemanticMappingUtils.getProperties;

public class SemanticMappingTransformer implements MappingTransformer {
    public final static Set<String> SUPPORTED_MODEL_ALGORITHM = Set.of(
        FunctionName.TEXT_EMBEDDING.name(),
        FunctionName.REMOTE.name(),
        FunctionName.SPARSE_ENCODING.name(),
        FunctionName.SPARSE_TOKENIZE.name()
    );
    public final static Set<String> SUPPORTED_REMOTE_MODEL_TYPES = Set.of(
        FunctionName.TEXT_EMBEDDING.name(),
        FunctionName.SPARSE_ENCODING.name(),
        FunctionName.SPARSE_TOKENIZE.name()
    );

    private final int FIELD_NAME_MAX_LENGTH = 512;

    private final MLCommonsClientAccessor mlClientAccessor;
    private final NamedXContentRegistry xContentRegistry;

    public SemanticMappingTransformer(
        @NonNull final MLCommonsClientAccessor mlClientAccessor,
        @NonNull final NamedXContentRegistry xContentRegistry
    ) {
        this.mlClientAccessor = mlClientAccessor;
        this.xContentRegistry = xContentRegistry;
    }

    /**
     * Add semantic info fields to the mapping.
     * @param mapping original mapping
     * e.g.
     *{
     *   "_doc": {
     *     "properties": {
     *       "semantic_field": {
     *         "model_id": "model_id",
     *         "type": "semantic"
     *       }
     *     }
     *   }
     * }
     *
     * It can be transformed to
     *{
     *   "_doc": {
     *     "properties": {
     *       "semantic_field": {
     *         "model_id": "model_id",
     *         "type": "semantic"
     *       },
     *       "semantic_field_semantic_info": {
     *         "properties": {
     *           "chunks": {
     *             "type": "nested",
     *             "properties": {
     *               "embedding": {
     *                 "type": "knn_vector",
     *                 "dimension": 768,
     *                 "method": {
     *                   "engine": "faiss",
     *                   "space_type": "l2",
     *                   "name": "hnsw",
     *                   "parameters": {}
     *                 }
     *               },
     *               "text": {
     *                 "type": "text"
     *               }
     *             }
     *           },
     *           "model": {
     *             "properties": {
     *               "id": {
     *                 "type": "text",
     *                 "index": false
     *               },
     *               "name": {
     *                 "type": "text",
     *                 "index": false
     *               },
     *               "type": {
     *                 "type": "text",
     *                 "index": false
     *               }
     *             }
     *           }
     *         }
     *       }
     *     }
     *   }
     * }
     * @param context context to help transform
     */

    @Override
    public void transform(final Map<String, Object> mapping, final TransformContext context, @NonNull final ActionListener<Void> listener) {
        try {
            final Map<String, Object> properties = getProperties(mapping);
            // If there is no property or its format is not valid we simply do nothing and rely on core to validate the
            // mappings and handle the error.
            if (properties == null) {
                listener.onResponse(null);
                return;
            }

            Map<String, Map<String, Object>> semanticFieldPathToConfigMap = new HashMap<>();
            String rootPath = Strings.EMPTY;
            SemanticMappingUtils.collectSemanticField(properties, rootPath, semanticFieldPathToConfigMap);

            validateSemanticFields(semanticFieldPathToConfigMap);

            fetchModelAndModifyMapping(semanticFieldPathToConfigMap, properties, listener);
        } catch (Exception e) {
            listener.onFailure(e);
        }

    }

    private void validateSemanticFields(@NonNull final Map<String, Map<String, Object>> semanticFieldPathToConfigMap) {
        final List<String> errors = new ArrayList<>();
        for (Map.Entry<String, Map<String, Object>> entry : semanticFieldPathToConfigMap.entrySet()) {
            final String semanticFieldPath = entry.getKey();
            final Map<String, Object> semanticFieldConfig = entry.getValue();
            errors.addAll(validateModelId(semanticFieldPath, semanticFieldConfig));
            errors.addAll(validateSemanticInfoFieldName(semanticFieldPath, semanticFieldConfig));
        }
        if (!errors.isEmpty()) {
            throw new IllegalArgumentException(String.join("; ", errors));
        }
    }

    private List<String> validateModelId(@NonNull final String semanticFieldPath, @NonNull final Map<String, Object> semanticFieldConfig) {
        final List<String> errors = new ArrayList<>();
        if (!semanticFieldConfig.containsKey(MODEL_ID)) {
            errors.add(MODEL_ID + " is required for the semantic field at " + semanticFieldPath);
        } else {
            Object modelId = semanticFieldConfig.get(MODEL_ID);
            if (!(modelId instanceof String modelIdStr && !modelIdStr.isEmpty())) {
                errors.add(MODEL_ID + " should be a non-empty string for the semantic field at " + semanticFieldPath);
            }
        }
        return errors;
    }

    private List<String> validateSemanticInfoFieldName(
        @NonNull final String semanticFieldPath,
        @NonNull final Map<String, Object> semanticFieldConfig
    ) {
        final List<String> errors = new ArrayList<>();
        if (semanticFieldConfig.containsKey(SEMANTIC_INFO_FIELD_NAME)) {
            final Object semanticInfoFieldName = semanticFieldConfig.get(SEMANTIC_INFO_FIELD_NAME);
            if (semanticInfoFieldName instanceof String semanticInfoFieldNameStr) {
                if (semanticInfoFieldNameStr.isEmpty()) {
                    errors.add(SEMANTIC_INFO_FIELD_NAME + " cannot be an empty string for the semantic field at " + semanticFieldPath);
                } else if (semanticInfoFieldNameStr.length() > FIELD_NAME_MAX_LENGTH) {
                    errors.add(
                        SEMANTIC_INFO_FIELD_NAME
                            + " should not be longer than "
                            + FIELD_NAME_MAX_LENGTH
                            + " characters for the semantic field at "
                            + semanticFieldPath
                    );
                }
            } else {
                errors.add(SEMANTIC_INFO_FIELD_NAME + " should be a non-empty string for the semantic field at " + semanticFieldPath);
            }
        }
        return errors;
    }

    private void fetchModelAndModifyMapping(
        @NonNull final Map<String, Map<String, Object>> semanticFieldPathToConfigMap,
        @NonNull final Map<String, Object> mappings,
        @NonNull final ActionListener<Void> listener
    ) {
        final Map<String, List<String>> modelIdToFieldPathMap = SemanticMappingUtils.extractModelIdToFieldPathMap(
            semanticFieldPathToConfigMap
        );
        if (modelIdToFieldPathMap.isEmpty()) {
            listener.onResponse(null);
        }
        final AtomicInteger counter = new AtomicInteger(modelIdToFieldPathMap.size());

        // we can have multiple semantic fields with different model ids and we should get model config for each model
        for (String modelId : modelIdToFieldPathMap.keySet()) {
            mlClientAccessor.getModel(modelId, new ActionListener<>() {
                @Override
                public void onResponse(MLModel mlModel) {
                    try {
                        synchronized (mappings) {
                            modifyMappings(mappings, mlModel, modelIdToFieldPathMap.get(modelId), semanticFieldPathToConfigMap, modelId);
                        }
                        if (counter.decrementAndGet() == 0) {
                            listener.onResponse(null);
                        }
                    } catch (Exception e) {
                        listener.onFailure(e);
                    }
                }

                @Override
                public void onFailure(Exception e) {
                    final String errorMessage = "Failed to transform mapping for semantic fields because failed to get "
                        + "the model info of the model id "
                        + modelId
                        + ". "
                        + e.getMessage();
                    listener.onFailure(new RuntimeException(errorMessage, e));
                }
            });
        }
    }

    private void modifyMappings(
        @NonNull final Map<String, Object> mappings,
        @NonNull final MLModel mlModel,
        @NonNull final List<String> fieldPaths,
        @NonNull final Map<String, Map<String, Object>> semanticFieldPathToConfigMap,
        @NonNull final String modelId
    ) {
        for (String fieldPath : fieldPaths) {
            final Map<String, Object> fieldConfig = semanticFieldPathToConfigMap.get(fieldPath);
            final String baseErrorMessage = "Failed to transform the mapping for the semantic field with model id "
                + modelId
                + " at path "
                + fieldPath
                + ". ";
            try {
                final Map<String, Object> semanticInfoConfig = createSemanticInfoField(mlModel);
                setSemanticInfoField(mappings, fieldPath, fieldConfig.get(SEMANTIC_INFO_FIELD_NAME), semanticInfoConfig);
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException(baseErrorMessage + e.getMessage(), e);
            } catch (Exception e) {
                throw new RuntimeException(baseErrorMessage + e.getMessage(), e);
            }
        }
    }

    @VisibleForTesting
    private Map<String, Object> createSemanticInfoField(final @NonNull MLModel modelConfig) throws IOException {
        final Map<String, Object> embeddingConfig = switch (modelConfig.getAlgorithm()) {
            case FunctionName.TEXT_EMBEDDING -> createEmbeddingConfigForTextEmbeddingModel(modelConfig);
            case FunctionName.SPARSE_ENCODING, FunctionName.SPARSE_TOKENIZE -> getBaseSparseEmbeddingConfig();
            case FunctionName.REMOTE -> createEmbeddingConfigForRemoteModel(modelConfig);
            default -> throw new IllegalArgumentException(
                modelConfig.getAlgorithm().name()
                    + " is not supported. The algorithm should be one of "
                    + String.join(", ", SUPPORTED_MODEL_ALGORITHM)
            );
        };

        return getBaseSemanticInfoConfig(embeddingConfig);
    }

    private Map<String, Object> createEmbeddingConfigForRemoteModel(@NonNull final MLModel modelConfig) throws IOException {
        final String modelType = modelConfig.getModelConfig().getModelType();
        final FunctionName modelTypeFunctionName;
        final String errMsg = "remote model type is not supported. It should be one of ["
            + String.join(", ", SUPPORTED_REMOTE_MODEL_TYPES)
            + "].";

        try {
            modelTypeFunctionName = FunctionName.from(modelType);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(getUnsupportedRemoteModelError());
        }

        return switch (modelTypeFunctionName) {
            case FunctionName.TEXT_EMBEDDING -> createEmbeddingConfigForTextEmbeddingModel(modelConfig);
            case FunctionName.SPARSE_ENCODING, FunctionName.SPARSE_TOKENIZE -> getBaseSparseEmbeddingConfig();
            default -> throw new IllegalArgumentException(getUnsupportedRemoteModelError());
        };
    }

    private String getUnsupportedRemoteModelError() {
        return "remote model type is not supported. It should be one of [" + String.join(", ", SUPPORTED_REMOTE_MODEL_TYPES) + "].";
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> createEmbeddingConfigForTextEmbeddingModel(@NonNull final MLModel modelConfig) throws IOException {
        final Map<String, Object> embeddingConfig = getBaseTextEmbeddingConfig();

        if (!(modelConfig.getModelConfig() instanceof TextEmbeddingModelConfig textEmbeddingModelConfig)) {
            throw new IllegalArgumentException("remote model is text embedding but model config is not a text embedding config.");
        }

        if (textEmbeddingModelConfig.getEmbeddingDimension() == null) {
            throw new IllegalArgumentException(
                "remote model is text embedding but embedding dimension is not defined " + "in the model config."
            );
        }
        embeddingConfig.put(KNN_VECTOR_DIMENSION_FIELD_NAME, textEmbeddingModelConfig.getEmbeddingDimension());

        final Map<String, Object> allConfigMap = MapperService.parseMapping(xContentRegistry, textEmbeddingModelConfig.getAllConfig());
        final Object spaceTypeObject = allConfigMap.get(KNN_VECTOR_METHOD_SPACE_TYPE_FIELD_NAME);
        if (!(spaceTypeObject instanceof String spaceTypeString)) {
            throw new IllegalArgumentException("space_type is not defined or not a string in the all config of the model.");
        }
        final Map<String, Object> methodConfig = (Map<String, Object>) embeddingConfig.get(KNN_VECTOR_METHOD_FIELD_NAME);
        methodConfig.put(KNN_VECTOR_METHOD_SPACE_TYPE_FIELD_NAME, spaceTypeString);

        return embeddingConfig;
    }

    @SuppressWarnings("unchecked")
    private void setSemanticInfoField(
        @NonNull final Map<String, Object> mappings,
        @NonNull final String fullPath,
        final Object userDefinedSemanticInfoFieldName,
        @NonNull final Map<String, Object> semanticInfoConfig
    ) {
        Map<String, Object> current = mappings;
        final String[] paths = fullPath.split("\\.");
        final String semanticInfoFieldName = userDefinedSemanticInfoFieldName == null
            ? paths[paths.length - 1] + SemanticFieldConstants.DEFAULT_SEMANTIC_INFO_FIELD_NAME_SUFFIX
            : (String) userDefinedSemanticInfoFieldName;
        paths[paths.length - 1] = semanticInfoFieldName;
        for (int i = 0; i < paths.length - 1; i++) {
            final Map<String, Object> interFieldConfig = (Map<String, Object>) current.get(paths[i]);
            current = (Map<String, Object>) interFieldConfig.get(MappingConstants.PROPERTIES);
        }

        // We simply set the whole semantic info config at the path of the semantic info. It is possible the config of
        // semantic info fields can be invalid, but we will not validate it here. We will rely on field mappers to
        // validate them when they parse the mappings.
        current.put(semanticInfoFieldName, semanticInfoConfig);
    }
}
