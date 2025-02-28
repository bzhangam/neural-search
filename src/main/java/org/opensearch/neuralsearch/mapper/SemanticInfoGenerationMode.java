/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.mapper;

import lombok.Getter;

@Getter
public enum SemanticInfoGenerationMode {
    ALWAYS("always"),
    DISABLED("disabled"),;

    private final String name;

    SemanticInfoGenerationMode(String name) {
        this.name = name;
    }

    public static SemanticInfoGenerationMode fromName(String name) {
        for (SemanticInfoGenerationMode e : SemanticInfoGenerationMode.values()) {
            if (e.getName().equals(name)) {
                return e;
            }
        }
        return null;
    }

    public static String availableValues() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (SemanticInfoGenerationMode e : SemanticInfoGenerationMode.values()) {
            sb.append(e.name).append(",");
        }
        sb.deleteCharAt(sb.length() - 1);
        sb.append("]");
        return sb.toString();
    }
}
