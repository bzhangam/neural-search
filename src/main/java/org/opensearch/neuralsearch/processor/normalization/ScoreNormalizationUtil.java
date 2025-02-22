/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.processor.normalization;

import lombok.extern.log4j.Log4j2;
import org.opensearch.neuralsearch.processor.explain.DocIdAtSearchShard;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * Collection of utility methods for score combination technique classes
 */
@Log4j2
class ScoreNormalizationUtil {
    private static final String PARAM_NAME_WEIGHTS = "weights";
    private static final float DELTA_FOR_SCORE_ASSERTION = 0.01f;

    /**
     * Validate config parameters for this technique
     * @param actualParams map of parameters in form of name-value
     * @param supportedParams collection of parameters that we should validate against, typically that's what is supported by exact technique
     */
    public void validateParams(final Map<String, Object> actualParams, final Set<String> supportedParams) {
        if (Objects.isNull(actualParams) || actualParams.isEmpty()) {
            return;
        }
        // check if only supported params are passed
        Optional<String> optionalNotSupportedParam = actualParams.keySet()
            .stream()
            .filter(paramName -> !supportedParams.contains(paramName))
            .findFirst();
        if (optionalNotSupportedParam.isPresent()) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "provided parameter for combination technique is not supported. supported parameters are [%s]",
                    String.join(",", supportedParams)
                )
            );
        }

        // check param types
        if (actualParams.keySet().stream().anyMatch(PARAM_NAME_WEIGHTS::equalsIgnoreCase)) {
            if (!(actualParams.get(PARAM_NAME_WEIGHTS) instanceof List)) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "parameter [%s] must be a collection of numbers", PARAM_NAME_WEIGHTS)
                );
            }
        }
    }

    /**
     * Sets a normalized score for a specific document at a specific subquery index
     *
     * @param normalizedScores map of document IDs to their list of scores
     * @param docIdAtSearchShard document ID
     * @param subQueryIndex index of the subquery
     * @param normalizedScore normalized score to set
     */
    public static void setNormalizedScore(
        Map<DocIdAtSearchShard, List<Float>> normalizedScores,
        DocIdAtSearchShard docIdAtSearchShard,
        int subQueryIndex,
        int numberOfSubQueries,
        float normalizedScore
    ) {
        List<Float> scores = normalizedScores.get(docIdAtSearchShard);
        if (Objects.isNull(scores)) {
            scores = new ArrayList<>(numberOfSubQueries);
            for (int i = 0; i < numberOfSubQueries; i++) {
                scores.add(0.0f);
            }
            normalizedScores.put(docIdAtSearchShard, scores);
        }
        scores.set(subQueryIndex, normalizedScore);
    }
}
