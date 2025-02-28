/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.common;

public class SemanticFieldConstants {
    public static String MODEL_ID = "model_id";
    public static String RAW_FIELD_TYPE = "raw_field_type";
    public static String SEMANTIC_INFO_FIELD_NAME = "semantic_info_field_name";
    public static String SEMANTIC_INFO_GENERATION_MODE = "semantic_info_generation_mode";

    public static String DEFAULT_SEMANTIC_INFO_FIELD_NAME_SUFFIX = "_semantic_info";

    public static String TYPE = "type";
    public static String PROPERTIES = "properties";

    public static class SemanticInfo {
        public static String CHUNKS = "chunks";
        public static String MODEL = "model";

        public static class Chunks {
            public static String TEXT = "test";
            public static String EMBEDDING = "embedding";
        }

        public static class ModelInfo {
            public static String TYPE = "type";
            public static String ID = "id";
            public static String NAME = "name";
        }
    }
}
