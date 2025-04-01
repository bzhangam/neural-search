/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.mappingtransformer;

import org.opensearch.neuralsearch.constants.MappingConstants;
import org.opensearch.neuralsearch.constants.SemanticFieldConstants;
import org.opensearch.neuralsearch.mapper.SemanticFieldMapper;
import reactor.util.annotation.NonNull;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A util class to help process mapping with semantic field
 */
public class SemanticMappingUtils {
    /**
     * It will recursively traverse the mapping to collect the full path to field config map for semantic fields
     * @param currentMapping current mapping
     * @param parentPath path of the parent object
     * @param semanticFieldPathToConfigMap path to field config map for semantic fields
     */
    @SuppressWarnings("unchecked")
    public static void collectSemanticField(
        @NonNull final Map<String, Object> currentMapping,
        @NonNull final String parentPath,
        @NonNull final Map<String, Map<String, Object>> semanticFieldPathToConfigMap
    ) {
        for (Map.Entry<String, Object> entry : currentMapping.entrySet()) {
            final String fieldName = entry.getKey();
            final Object fieldConfig = entry.getValue();

            // Build the full path for the current field
            final String fullPath = parentPath.isEmpty() ? fieldName : parentPath + "." + fieldName;

            if (fieldConfig instanceof Map) {
                final Map<String, Object> fieldConfigMap = (Map<String, Object>) fieldConfig;

                if (isSemanticField(fieldConfigMap)) {
                    semanticFieldPathToConfigMap.put(fullPath, fieldConfigMap);
                }

                // If it's an object field, recurse into the sub fields
                if (fieldConfigMap.containsKey(MappingConstants.PROPERTIES)) {
                    collectSemanticField(
                        (Map<String, Object>) fieldConfigMap.get(MappingConstants.PROPERTIES),
                        fullPath,
                        semanticFieldPathToConfigMap
                    );
                }
            }
        }
    }

    private static boolean isSemanticField(Map<String, Object> fieldConfigMap) {
        return fieldConfigMap.containsKey(MappingConstants.TYPE)
            && SemanticFieldMapper.CONTENT_TYPE.equals(fieldConfigMap.get(MappingConstants.TYPE));
    }

    /**
     * In mapping there can be multiple semantic fields with the same model id. This function can help build the model
     * id to the path of semantic fields map. In this way we can know how many unique model ids we have and pull the
     * model info for them.
     * @param semanticFieldPathToConfigMap path to semantic field config map
     * @return model id to paths of semantic fields map
     */
    public static Map<String, List<String>> extractModelIdToFieldPathMap(
        @NonNull final Map<String, Map<String, Object>> semanticFieldPathToConfigMap
    ) {
        final Map<String, List<String>> modelIdToFieldPathMap = new HashMap<>();
        for (Map.Entry<String, Map<String, Object>> entry : semanticFieldPathToConfigMap.entrySet()) {
            final String fullPath = entry.getKey();
            final Map<String, Object> fieldConfigMap = entry.getValue();
            final String modelIdStr = getModelId(fieldConfigMap, fullPath);
            if (modelIdToFieldPathMap.containsKey(modelIdStr)) {
                modelIdToFieldPathMap.get(modelIdStr).add(fullPath);
            } else {
                modelIdToFieldPathMap.put(modelIdStr, new ArrayList<>());
                modelIdToFieldPathMap.get(modelIdStr).add(fullPath);
            }
        }
        return modelIdToFieldPathMap;
    }

    private static String getModelId(@NonNull final Map<String, Object> fieldConfigMap, @NonNull final String fullPath) {
        final Object modelId = fieldConfigMap.get(SemanticFieldConstants.MODEL_ID);
        if (!(modelId instanceof String)) {
            throw new IllegalArgumentException("Model ID is a required string value for semantic field: " + fullPath);
        }
        return (String) modelId;
    }
}
