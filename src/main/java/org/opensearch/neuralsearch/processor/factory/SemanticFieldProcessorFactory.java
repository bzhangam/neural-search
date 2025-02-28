/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.processor.factory;

import joptsimple.internal.Strings;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.env.Environment;
import org.opensearch.index.analysis.AnalysisRegistry;
import org.opensearch.ingest.AbstractBatchingProcessor;
import org.opensearch.neuralsearch.common.SemanticFieldConstants;
import org.opensearch.neuralsearch.mapper.SemanticInfoGenerationMode;
import org.opensearch.neuralsearch.ml.MLCommonsClientAccessor;
import org.opensearch.neuralsearch.processor.semantic.SemanticFieldProcessor;
import org.opensearch.neuralsearch.util.SemanticMappingUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opensearch.ingest.ConfigurationUtils.readOptionalMap;
import static org.opensearch.neuralsearch.util.SemanticMappingUtils.getSemanticInfoGenerationMode;

/**
 * Factory for semantic field internal ingest processors. Instantiates processor based on index configuration.
 */
public final class SemanticFieldProcessorFactory extends AbstractBatchingProcessor.Factory {
    public static final String PROCESSOR_FACTORY_TYPE = "internal_semantic_field";
    public static final String INDEX_MAPPING_FIELD = "index_mapping";

    private final MLCommonsClientAccessor mlClientAccessor;

    private final Environment environment;

    private final ClusterService clusterService;
    private final AnalysisRegistry analysisRegistry;

    public SemanticFieldProcessorFactory(
        final MLCommonsClientAccessor mlClientAccessor,
        final Environment environment,
        final ClusterService clusterService,
        AnalysisRegistry analysisRegistry
    ) {
        super(PROCESSOR_FACTORY_TYPE);
        this.mlClientAccessor = mlClientAccessor;
        this.environment = environment;
        this.clusterService = clusterService;
        this.analysisRegistry = analysisRegistry;
    }

    @Override
    public boolean isInternal() {
        return true;
    }

    @Override
    protected AbstractBatchingProcessor newProcessor(String tag, String description, int batchSize, Map<String, Object> config) {
        Map<String, Object> mapping = readOptionalMap(PROCESSOR_FACTORY_TYPE, tag, config, INDEX_MAPPING_FIELD);
        if (description == null) {
            description = "Index based ingest processor for the semantic field.";
        }
        // path -> field config info
        // product_list.product_description -> {"type": "semantic_text", "model_id": "123", "isParentNestedType": true}
        Map<String, Map<String, Object>> semanticFieldPathToConfigMap = new HashMap<>();
        String rootPath = Strings.EMPTY;
        SemanticMappingUtils.collectSemanticField(
            (Map<String, Object>) mapping.get(SemanticFieldConstants.PROPERTIES),
            rootPath,
            semanticFieldPathToConfigMap
        );

        filterSemanticInfoGenerationDisabledFields(semanticFieldPathToConfigMap);

        if (semanticFieldPathToConfigMap.isEmpty()) {
            return null;
        }

        return new SemanticFieldProcessor(
            tag,
            description,
            batchSize,
            semanticFieldPathToConfigMap,
            analysisRegistry,
            mlClientAccessor,
            environment,
            clusterService
        );
    }

    private void filterSemanticInfoGenerationDisabledFields(Map<String, Map<String, Object>> semanticFieldPathToConfigMap) {
        List<String> keysToRemove = new ArrayList<>();
        for (Map.Entry<String, Map<String, Object>> entry : semanticFieldPathToConfigMap.entrySet()) {
            SemanticInfoGenerationMode mode = getSemanticInfoGenerationMode(entry.getValue(), entry.getKey());
            if (SemanticInfoGenerationMode.DISABLED.equals(mode)) {
                keysToRemove.add(entry.getKey());
            }
        }
        keysToRemove.forEach(semanticFieldPathToConfigMap::remove);
    }
}
