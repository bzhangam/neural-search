/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.util;

import org.opensearch.neuralsearch.constants.MappingConstants;
import org.opensearch.neuralsearch.constants.SemanticFieldConstants;
import org.opensearch.neuralsearch.mapper.SemanticFieldMapper;
import reactor.util.annotation.NonNull;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.opensearch.neuralsearch.constants.MappingConstants.DOC;
import static org.opensearch.neuralsearch.constants.MappingConstants.PATH_SEPARATOR;
import static org.opensearch.neuralsearch.constants.MappingConstants.PROPERTIES;

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
            final String fullPath = parentPath.isEmpty() ? fieldName : parentPath + PATH_SEPARATOR + fieldName;

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
            if (modelIdToFieldPathMap.containsKey(modelIdStr) == false) {
                modelIdToFieldPathMap.put(modelIdStr, new ArrayList<>());
            }
            modelIdToFieldPathMap.get(modelIdStr).add(fullPath);
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

    /**
     * Collect unique model ids defined in the semantic fields
     *
     * @param semanticFieldPathToConfigMap path to config of semantic fields defined in the index mapping
     * @return unique model ids defined in the semantic fields
     */
    public static Set<String> getUniqueModelIds(@NonNull final Map<String, Map<String, Object>> semanticFieldPathToConfigMap) {
        final Set<String> modelIds = new HashSet<>();
        for (Map.Entry<String, Map<String, Object>> entry : semanticFieldPathToConfigMap.entrySet()) {
            final Map<String, Object> fieldConfigMap = entry.getValue();
            modelIds.add(getModelId(fieldConfigMap, entry.getKey()));
        }
        return modelIds;
    }

    /**
     * Help extract the properties from a mapping
     * @param mapping index mapping
     * @return properties of the mapping
     */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> getProperties(Map<String, Object> mapping) {
        if (mapping == null) {
            return null;
        }
        // Actions like create index and legacy create/update index template will have the mapping properties under a
        // _doc key. Other actions like update mapping and create/update index template will not have the _doc layer.
        if (mapping.containsKey(DOC) && mapping.get(DOC) instanceof Map) {
            Map<String, Object> doc = (Map<String, Object>) mapping.get(DOC);
            if (doc.containsKey(PROPERTIES) && doc.get(PROPERTIES) instanceof Map) {
                return (Map<String, Object>) doc.get(PROPERTIES);
            } else {
                return null;
            }
        } else if (mapping.containsKey(PROPERTIES) && mapping.get(PROPERTIES) instanceof Map) {
            return (Map<String, Object>) mapping.get(PROPERTIES);
        } else {
            return null;
        }
    }

    /**
     * Get the config of the target field from the index mapping by its path.
     * @param mapping index mapping
     * @param path path to the target field
     * @return The config of the target field in the index mapping.
     */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> getFieldConfigByPath(final Map<String, Object> mapping, final String path) {
        final String[] paths = path.split("\\.");
        Map<String, Object> currentMapping = getProperties(mapping);
        for (int i = 0; i < paths.length; i++) {
            if (currentMapping == null) {
                return null;
            }
            final Object temp = currentMapping.get(paths[i]);
            if (temp instanceof Map) {
                currentMapping = (Map<String, Object>) temp;
                // handle the object field in the path
                if (i < paths.length - 1 && currentMapping.containsKey(PROPERTIES)) {
                    currentMapping = (Map<String, Object>) currentMapping.get(PROPERTIES);
                }
            } else {
                return null;
            }
        }
        return currentMapping;
    }
}
