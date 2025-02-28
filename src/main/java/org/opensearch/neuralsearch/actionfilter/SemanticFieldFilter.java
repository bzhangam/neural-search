/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.actionfilter;

import joptsimple.internal.Strings;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.admin.indices.create.CreateIndexRequest;
import org.opensearch.action.support.ActionFilter;
import org.opensearch.action.support.ActionFilterChain;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.action.ActionResponse;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.model.TextEmbeddingModelConfig;
import org.opensearch.neuralsearch.common.SemanticFieldConstants;
import org.opensearch.neuralsearch.ml.MLCommonsClientAccessor;
import org.opensearch.neuralsearch.util.SemanticMappingUtils;
import org.opensearch.tasks.Task;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SemanticFieldFilter implements ActionFilter {
    private final MLCommonsClientAccessor mlClientAccessor;

    public SemanticFieldFilter(MLCommonsClientAccessor mlClientAccessor) {
        this.mlClientAccessor = mlClientAccessor;
    }

    @Override
    public int order() {
        return 1;
    }

    @Override
    public <Request extends ActionRequest, Response extends ActionResponse> void apply(
        Task task,
        String action,
        Request request,
        ActionListener<Response> listener,
        ActionFilterChain<Request, Response> chain
    ) {
        if (request instanceof CreateIndexRequest) {
            processCreateIndexRequest(task, action, request, listener, chain);
        } else {
            chain.proceed(task, action, request, listener);
        }

    }

    private <Request extends ActionRequest, Response extends ActionResponse> void processCreateIndexRequest(
        Task task,
        String action,
        Request request,
        ActionListener<Response> listener,
        ActionFilterChain<Request, Response> chain
    ) {
        try {
            Map<String, Object> mappings = parseMappings((CreateIndexRequest) request);

            // List to hold full paths to all semantic fields and their corresponding model IDs
            Map<String, Map<String, Object>> semanticFieldPathToConfigMap = new HashMap<>();
            String rootPath = Strings.EMPTY;
            SemanticMappingUtils.collectSemanticField(mappings, rootPath, semanticFieldPathToConfigMap);

            if (semanticFieldPathToConfigMap.isEmpty()) {
                chain.proceed(task, action, request, listener);
            } else {
                fetchModelAndModifyMapping(task, action, request, listener, chain, semanticFieldPathToConfigMap, mappings);
            }
        } catch (IOException e) {
            listener.onFailure(new RuntimeException("Failed to process index mapping", e));
        }
    }

    private Map<String, Object> parseMappings(CreateIndexRequest request) throws IOException {
        Map<String, Object> mapping = request.mappings() != null
            ? XContentHelper.convertToMap(XContentType.JSON.xContent(), request.mappings(), false)
            : Map.of();
        Object mappingDoc = mapping.get(MapperService.SINGLE_MAPPING_NAME);
        if (mappingDoc instanceof Map) {
            Object properties = ((Map) mappingDoc).get(SemanticFieldConstants.PROPERTIES);
            if (properties != null && properties instanceof Map) {
                return (Map<String, Object>) properties;
            }
        }
        // Rely on field mapper to throw the validation error when the index mapping is invalid.
        return new HashMap<>();
    }

    private void setMappings(CreateIndexRequest request, Map<String, Object> mappings) {
        Map<String, Object> mappingDoc = new HashMap<>();
        mappingDoc.put(SemanticFieldConstants.PROPERTIES, mappings);
        request.mapping(mappingDoc);
    }

    private <Request extends ActionRequest, Response extends ActionResponse> void fetchModelAndModifyMapping(
        Task task,
        String action,
        Request request,
        ActionListener<Response> listener,
        ActionFilterChain<Request, Response> chain,
        Map<String, Map<String, Object>> semanticFieldPathToConfigMap,
        Map<String, Object> mappings
    ) {

        Map<String, List<String>> idToFieldPathMap = SemanticMappingUtils.extractModelIdToFieldPathMap(semanticFieldPathToConfigMap);

        final int[] counter = { idToFieldPathMap.size() };
        for (String modelId : idToFieldPathMap.keySet()) {
            mlClientAccessor.getModel(modelId, new ActionListener<MLModel>() {
                @Override
                public void onResponse(MLModel mlModel) {
                    try {
                        modifyMappings(mappings, mlModel, idToFieldPathMap.get(modelId), semanticFieldPathToConfigMap);
                        if (--counter[0] == 0) {
                            setMappings((CreateIndexRequest) request, mappings);
                            chain.proceed(task, action, request, listener);
                        }
                    } catch (IOException e) {
                        listener.onFailure(new RuntimeException("Failed to modify index mapping", e));
                    }
                }

                @Override
                public void onFailure(Exception e) {
                    listener.onFailure(new RuntimeException("Failed to create semantic info field.", e));
                }
            });
        }

    }

    private void modifyMappings(
        Map<String, Object> mappings,
        MLModel mlModel,
        List<String> fieldPaths,
        Map<String, Map<String, Object>> semanticFieldPathToConfigMap
    ) throws IOException {
        for (String fieldPath : fieldPaths) {
            Map<String, Object> fieldConfig = semanticFieldPathToConfigMap.get(fieldPath);
            Map<String, Object> semanticInfoConfig = createSemanticInfoField(fieldConfig, mlModel);

            setSemanticInfoField(mappings, fieldPath, fieldConfig.get(SemanticFieldConstants.SEMANTIC_INFO_FIELD_NAME), semanticInfoConfig);
        }
    }

    // Helper method to create the semantic info field
    private Map<String, Object> createSemanticInfoField(Map<String, Object> fieldConfig, MLModel modelConfig) {

        Map<String, Object> embeddingConfig = new HashMap<>();
        if (FunctionName.TEXT_EMBEDDING.equals(modelConfig.getAlgorithm())) {
            TextEmbeddingModelConfig textEmbeddingModelConfig = (TextEmbeddingModelConfig) modelConfig.getModelConfig();
            embeddingConfig = Map.of(
                "type",
                "knn_vector",
                "dimension",
                textEmbeddingModelConfig.getEmbeddingDimension(),
                "method",
                Map.of("engine", "faiss", "name", "hnsw", "space_type", "l2")
            );
        }
        ;

        Map<String, Object> semanticInfoField = new HashMap<>();
        semanticInfoField.put(
            "properties",
            Map.of(
                "chunks",
                Map.of("type", "nested", "properties", Map.of("text", Map.of("type", "text"), "embedding", embeddingConfig)),
                "model",
                Map.of(
                    "properties",
                    Map.of(
                        "id",
                        Map.of("type", "text", "index", "false"),
                        "type",
                        Map.of("type", "text", "index", "false"),
                        "name",
                        Map.of("type", "text", "index", "false")
                    )
                )
            )
        );
        return semanticInfoField;
    }

    private void setSemanticInfoField(
        Map<String, Object> mappings,
        String fullPath,
        Object userDefinedSemanticInfoFieldName,
        Map<String, Object> semanticInfoConfig
    ) {
        if (userDefinedSemanticInfoFieldName != null && !(userDefinedSemanticInfoFieldName instanceof String)) {
            throw new IllegalArgumentException(
                SemanticFieldConstants.SEMANTIC_INFO_FIELD_NAME + " should be a string semantic field at: " + fullPath
            );
        }

        Map<String, Object> current = mappings;
        String[] paths = fullPath.split("\\.");
        String semanticInfoFieldName = userDefinedSemanticInfoFieldName == null
            ? paths[paths.length - 1] + SemanticFieldConstants.DEFAULT_SEMANTIC_INFO_FIELD_NAME_SUFFIX
            : (String) userDefinedSemanticInfoFieldName;
        paths[paths.length - 1] = semanticInfoFieldName;
        for (int i = 0; i < paths.length - 1; i++) {
            Map<String, Object> interFieldConfig = (Map<String, Object>) current.get(paths[i]);
            current = (Map<String, Object>) interFieldConfig.get(SemanticFieldConstants.PROPERTIES);
        }
        if (current.containsKey(paths[paths.length - 1])) {
            throw new IllegalArgumentException(
                "Field "
                    + semanticInfoFieldName
                    + " already exists in path "
                    + String.join(".", paths)
                    + ". Semantic field cannot auto generate semantic info field. Please rename it or define a custom semantic_info_field_name to avoid the conflict."
            );
        }
        current.put(semanticInfoFieldName, semanticInfoConfig);
    }
}
