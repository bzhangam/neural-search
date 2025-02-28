/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.util;

import org.opensearch.neuralsearch.common.SemanticFieldConstants;
import org.opensearch.neuralsearch.mapper.SemanticFieldMapper;
import org.opensearch.neuralsearch.mapper.SemanticInfoGenerationMode;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class SemanticMappingUtils {

    // Recursive method to traverse the mapping and collect paths of semantic fields and field config
    public static void collectSemanticField(
        Map<String, Object> currentMapping,
        String parentPath,
        Map<String, Map<String, Object>> semanticFieldPathToConfigMapping
    ) {
        for (Map.Entry<String, Object> entry : currentMapping.entrySet()) {
            String fieldName = entry.getKey();
            Object fieldConfig = entry.getValue();

            // Build the full path for the current field
            String fullPath = parentPath.isEmpty() ? fieldName : parentPath + "." + fieldName;

            if (fieldConfig instanceof Map) {
                Map<String, Object> fieldConfigMap = (Map<String, Object>) fieldConfig;
                // If it's a semantic field, store its path and field config
                if (isSemanticField(fieldConfigMap)) {
                    semanticFieldPathToConfigMapping.put(fullPath, fieldConfigMap);
                }

                // If it's an object field, recurse into the sub fields
                if (fieldConfigMap.containsKey(SemanticFieldConstants.PROPERTIES)) {
                    collectSemanticField(
                        (Map<String, Object>) fieldConfigMap.get(SemanticFieldConstants.PROPERTIES),
                        fullPath,
                        semanticFieldPathToConfigMapping
                    );
                }
            }
        }
    }

    // Helper method to check if a field is of type semantic
    private static boolean isSemanticField(Map<String, Object> fieldConfigMap) {
        // Logic to check if fieldConfig corresponds to a semantic field type
        return fieldConfigMap.containsKey(SemanticFieldConstants.TYPE)
            && SemanticFieldMapper.CONTENT_TYPE.equals(fieldConfigMap.get(SemanticFieldConstants.TYPE));
    }

    public static Set<String> getUniqueModelIds(Map<String, Map<String, Object>> semanticFieldPathToConfigMap) {
        Set<String> modelIds = new HashSet<>();
        for (Map.Entry<String, Map<String, Object>> entry : semanticFieldPathToConfigMap.entrySet()) {
            Map<String, Object> fieldConfigMap = entry.getValue();
            modelIds.add(getModelId(fieldConfigMap, entry.getKey()));
        }
        return modelIds;
    }

    private static String getModelId(Map<String, Object> fieldConfigMap, String fullPath) {
        Object modelId = fieldConfigMap.get(SemanticFieldConstants.MODEL_ID);
        if (!(modelId instanceof String)) {
            throw new IllegalArgumentException("Model ID is a required string value for semantic field: " + fullPath);
        }
        return (String) modelId;
    }

    public static SemanticInfoGenerationMode getSemanticInfoGenerationMode(Map<String, Object> fieldConfigMap, String fullPath) {
        Object modeName = fieldConfigMap.get(SemanticFieldConstants.SEMANTIC_INFO_GENERATION_MODE);
        if (modeName == null) {
            return SemanticInfoGenerationMode.ALWAYS;
        }
        String err = String.format(
            "semantic_info_generation_mode should be one of %s for semantic field %s.",
            SemanticInfoGenerationMode.availableValues(),
            fullPath
        );
        if (!(modeName instanceof String)) {
            throw new IllegalArgumentException(err);
        }
        SemanticInfoGenerationMode mode = SemanticInfoGenerationMode.fromName((String) modeName);
        if (mode == null) {
            throw new IllegalArgumentException(err);
        }
        return mode;
    }

    // Helper method to extract the model_id from the field config
    public static Map<String, List<String>> extractModelIdToFieldPathMap(Map<String, Map<String, Object>> semanticFieldPathToConfigMap) {
        Map<String, List<String>> idToFieldPathMap = new HashMap<>();
        for (Map.Entry<String, Map<String, Object>> entry : semanticFieldPathToConfigMap.entrySet()) {
            String fullPath = entry.getKey();
            Map<String, Object> fieldConfigMap = entry.getValue();
            String modelIdStr = getModelId(fieldConfigMap, fullPath);
            if (idToFieldPathMap.containsKey(modelIdStr)) {
                idToFieldPathMap.get(modelIdStr).add(fullPath);
            } else {
                idToFieldPathMap.put(modelIdStr, new ArrayList<>());
                idToFieldPathMap.get(modelIdStr).add(fullPath);
            }
        }
        return idToFieldPathMap;
    }

}
