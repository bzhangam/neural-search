/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.processor.factory;

import org.opensearch.cluster.service.ClusterService;
import org.opensearch.env.Environment;
import org.opensearch.ingest.AbstractBatchingProcessor;
import org.opensearch.neuralsearch.mapper.SemanticTextFieldMapper;
import org.opensearch.neuralsearch.ml.MLCommonsClientAccessor;
import org.opensearch.neuralsearch.processor.semantic.SemanticFieldProcessor;

import java.util.HashMap;
import java.util.Map;

import static org.opensearch.ingest.ConfigurationUtils.readOptionalMap;

/**
 * Factory for semantic field internal ingest processors. Instantiates processor based on index configuration.
 */
public final class SemanticFieldProcessorFactory extends AbstractBatchingProcessor.Factory {
    public static final String PROCESSOR_FACTORY_TYPE = "internal_semantic_field";
    public static final String INDEX_MAPPING_FIELD = "index_mapping";

    private final MLCommonsClientAccessor clientAccessor;

    private final Environment environment;

    private final ClusterService clusterService;

    public SemanticFieldProcessorFactory(
        final MLCommonsClientAccessor clientAccessor,
        final Environment environment,
        final ClusterService clusterService
    ) {
        super(PROCESSOR_FACTORY_TYPE);
        this.clientAccessor = clientAccessor;
        this.environment = environment;
        this.clusterService = clusterService;
    }

    @Override
    public boolean isInternal() {
        return true;
    }

    @Override
    protected AbstractBatchingProcessor newProcessor(String tag, String description, int batchSize, Map<String, Object> config) {
        Map<String, Object> mapping = readOptionalMap(PROCESSOR_FACTORY_TYPE, tag, config, INDEX_MAPPING_FIELD);
        if (description == null) {
            description = "Internal processor for semantic field";
        }

        // TODO: Go through the mapping to see if we have semantic field and collect
        // path -> field config info
        // product_list.product_description -> {"type": "semantic_text", "model_id": "123", "isParentNestedType": true}

        Map<String, Object> pathToFieldConfig = new HashMap<>();

        collectSemanticFields((Map<String, Object>) mapping.get("properties"), pathToFieldConfig, "", false, true);

        // TODO: If find semantic text field we should pull the model config and pass it to processor so that we don't
        // need to keep pulling it everytime we process a doc. At the same time we also can check if the model is valid
        // earlier.
        if (pathToFieldConfig.isEmpty()) {
            return null;
        }

        return new SemanticFieldProcessor(tag, description, batchSize, pathToFieldConfig, clientAccessor, environment, clusterService);
    }

    private void collectSemanticFields(
        Map<String, Object> mapping,
        Map<String, Object> pathToFieldConfig,
        String parentPath,
        boolean isNestedObject,
        boolean isRoot
    ) {
        for (Map.Entry<String, Object> entry : mapping.entrySet()) {
            String fieldName = entry.getKey();
            String fullPath = isRoot ? fieldName : parentPath + "." + fieldName;
            Object fieldConfig = entry.getValue();
            if (fieldConfig instanceof Map) {
                String fieldType = (String) ((Map<?, ?>) fieldConfig).get("type");
                if (SemanticTextFieldMapper.CONTENT_TYPE.equals(fieldType)) {
                    ((Map<String, Object>) fieldConfig).put("isNestedObject", isNestedObject);
                    pathToFieldConfig.put(fullPath, fieldConfig);
                }
                Object properties = ((Map<?, ?>) fieldConfig).get("properties");
                if (properties instanceof Map) {
                    collectSemanticFields((Map<String, Object>) properties, pathToFieldConfig, fullPath, "nested".equals(fieldName), false);
                }
            }
        }
    }
}
