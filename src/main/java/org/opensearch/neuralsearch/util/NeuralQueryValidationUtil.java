/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.util;

import lombok.NonNull;
import org.apache.commons.lang.StringUtils;
import org.opensearch.neuralsearch.query.NeuralQueryBuilder;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import static org.opensearch.knn.index.query.KNNQueryBuilder.EXPAND_NESTED_FIELD;
import static org.opensearch.knn.index.query.KNNQueryBuilder.FILTER_FIELD;
import static org.opensearch.knn.index.query.KNNQueryBuilder.K_FIELD;
import static org.opensearch.knn.index.query.KNNQueryBuilder.MAX_DISTANCE_FIELD;
import static org.opensearch.knn.index.query.KNNQueryBuilder.METHOD_PARAMS_FIELD;
import static org.opensearch.knn.index.query.KNNQueryBuilder.MIN_SCORE_FIELD;
import static org.opensearch.knn.index.query.KNNQueryBuilder.RESCORE_FIELD;
import static org.opensearch.neuralsearch.common.MinClusterVersionUtil.isClusterOnOrAfterMinReqVersionForDefaultDenseModelIdSupport;
import static org.opensearch.neuralsearch.query.NeuralQueryBuilder.MODEL_ID_FIELD;
import static org.opensearch.neuralsearch.query.NeuralQueryBuilder.QUERY_IMAGE_FIELD;
import static org.opensearch.neuralsearch.query.NeuralQueryBuilder.QUERY_TEXT_FIELD;
import static org.opensearch.neuralsearch.query.NeuralQueryBuilder.QUERY_TOKENS_FIELD;

public class NeuralQueryValidationUtil {
    public static List<String> validateNeuralQueryForKnn(@NonNull final NeuralQueryBuilder queryBuilder) {
        final List<String> errors = validateNeuralQueryForDenseCommonRules(queryBuilder);

        if (!isClusterOnOrAfterMinReqVersionForDefaultDenseModelIdSupport()) {
            if (queryBuilder.modelId() == null) {
                errors.add(String.format(Locale.ROOT, "%s must be provided.", MODEL_ID_FIELD.getPreferredName()));
            }
        }

        return errors;
    }

    private static int countKnnQueryTypes(NeuralQueryBuilder queryBuilder) {
        int counter = 0;
        if (queryBuilder.k() != null) {
            counter++;
        }
        if (queryBuilder.maxDistance() != null) {
            counter++;
        }
        if (queryBuilder.minScore() != null) {
            counter++;
        }
        return counter;
    }

    public static List<String> validateNeuralQueryForSemanticDense(@NonNull final NeuralQueryBuilder queryBuilder) {
        final List<String> errors = validateNeuralQueryForDenseCommonRules(queryBuilder);

        if (queryBuilder.queryTokensMapSupplier() != null) {
            errors.add(
                String.format(
                    Locale.ROOT,
                    "Target field is a semantic field using a dense model. %s is not supported since it is for the sparse model.",
                    QUERY_TOKENS_FIELD.getPreferredName()
                )
            );
        }

        return errors;
    }

    private static List<String> validateNeuralQueryForDenseCommonRules(@NonNull final NeuralQueryBuilder queryBuilder) {
        final List<String> errors = new ArrayList<>();

        if (StringUtils.isBlank(queryBuilder.queryText()) && StringUtils.isBlank(queryBuilder.queryImage())) {
            errors.add(
                String.format(
                    Locale.ROOT,
                    "Either %s or %s must be provided.",
                    QUERY_TEXT_FIELD.getPreferredName(),
                    QUERY_IMAGE_FIELD.getPreferredName()
                )
            );
        }

        int knnQueryTypes = countKnnQueryTypes(queryBuilder);
        if (knnQueryTypes > 1) {
            errors.add(
                String.format(
                    Locale.ROOT,
                    "Only one of %s, %s, or %s can be provided",
                    K_FIELD.getPreferredName(),
                    MAX_DISTANCE_FIELD.getPreferredName(),
                    MIN_SCORE_FIELD.getPreferredName()
                )
            );
        } else if (knnQueryTypes == 0) {
            queryBuilder.k(NeuralQueryBuilder.DEFAULT_K);
        }

        if (StringUtils.EMPTY.equals(queryBuilder.modelId())) {
            errors.add(String.format(Locale.ROOT, "%s field can not be empty", MODEL_ID_FIELD.getPreferredName()));
        }
        return errors;
    }

    public static List<String> validateNeuralQueryForSemanticSparse(@NonNull final NeuralQueryBuilder queryBuilder) {
        final List<String> errors = new ArrayList<>();

        if (queryBuilder.queryTokensMapSupplier() == null && queryBuilder.queryText() == null) {
            errors.add(
                String.format(
                    Locale.ROOT,
                    "Either %s or %s must be provided",
                    QUERY_TEXT_FIELD.getPreferredName(),
                    QUERY_TOKENS_FIELD.getPreferredName()
                )
            );
        } else if (queryBuilder.queryTokensMapSupplier() == null && StringUtils.EMPTY.equals(queryBuilder.queryText())) {
            errors.add(String.format(Locale.ROOT, "%s field can not be empty", QUERY_TEXT_FIELD.getPreferredName()));
        }

        if (StringUtils.EMPTY.equals(queryBuilder.modelId())) {
            errors.add(String.format(Locale.ROOT, "%s field can not be empty", MODEL_ID_FIELD.getPreferredName()));
        }

        final Set<String> fieldsOnlySupportedByDenseModel = getFieldsOnlySupportedByDenseModel(queryBuilder);

        if (!fieldsOnlySupportedByDenseModel.isEmpty()) {
            errors.add(
                String.format(
                    Locale.ROOT,
                    "Target field is a semantic field using a sparse model. [%s] are not supported since they are for the dense model.",
                    String.join(", ", fieldsOnlySupportedByDenseModel)
                )
            );
        }

        return errors;
    }

    private static Set<String> getFieldsOnlySupportedByDenseModel(NeuralQueryBuilder queryBuilder) {
        final Set<String> fieldsOnlySupportedByDenseModel = new HashSet<>();
        if (queryBuilder.queryImage() != null) {
            fieldsOnlySupportedByDenseModel.add(QUERY_IMAGE_FIELD.getPreferredName());
        }

        if (queryBuilder.k() != null) {
            fieldsOnlySupportedByDenseModel.add(K_FIELD.getPreferredName());
        }

        if (queryBuilder.maxDistance() != null) {
            fieldsOnlySupportedByDenseModel.add(MAX_DISTANCE_FIELD.getPreferredName());
        }

        if (queryBuilder.minScore() != null) {
            fieldsOnlySupportedByDenseModel.add(MIN_SCORE_FIELD.getPreferredName());
        }

        if (queryBuilder.expandNested() != null) {
            fieldsOnlySupportedByDenseModel.add(EXPAND_NESTED_FIELD.getPreferredName());
        }

        if (queryBuilder.filter() != null) {
            fieldsOnlySupportedByDenseModel.add(FILTER_FIELD.getPreferredName());
        }

        if (queryBuilder.methodParameters() != null) {
            fieldsOnlySupportedByDenseModel.add(METHOD_PARAMS_FIELD.getPreferredName());
        }

        if (queryBuilder.rescoreContext() != null) {
            fieldsOnlySupportedByDenseModel.add(RESCORE_FIELD.getPreferredName());
        }
        return fieldsOnlySupportedByDenseModel;
    }
}
