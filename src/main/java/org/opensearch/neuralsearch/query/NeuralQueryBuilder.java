/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.query;

import static org.opensearch.knn.index.query.KNNQueryBuilder.EXPAND_NESTED_FIELD;
import static org.opensearch.knn.index.query.KNNQueryBuilder.FILTER_FIELD;
import static org.opensearch.knn.index.query.KNNQueryBuilder.MAX_DISTANCE_FIELD;
import static org.opensearch.knn.index.query.KNNQueryBuilder.METHOD_PARAMS_FIELD;
import static org.opensearch.knn.index.query.KNNQueryBuilder.MIN_SCORE_FIELD;
import static org.opensearch.knn.index.query.KNNQueryBuilder.RESCORE_FIELD;
import static org.opensearch.neuralsearch.common.MinClusterVersionUtil.isClusterOnOrAfterMinReqVersion;
import static org.opensearch.neuralsearch.common.MinClusterVersionUtil.isClusterOnOrAfterMinReqVersionForDefaultDenseModelIdSupport;
import static org.opensearch.neuralsearch.common.MinClusterVersionUtil.isClusterOnOrAfterMinReqVersionForRadialSearch;
import static org.opensearch.neuralsearch.common.VectorUtil.vectorAsListToArray;
import static org.opensearch.neuralsearch.constants.MappingConstants.PATH_SEPARATOR;
import static org.opensearch.neuralsearch.constants.MappingConstants.TYPE;
import static org.opensearch.neuralsearch.constants.SemanticFieldConstants.DEFAULT_SEMANTIC_INFO_FIELD_NAME_SUFFIX;
import static org.opensearch.neuralsearch.constants.SemanticFieldConstants.MODEL_ID;
import static org.opensearch.neuralsearch.constants.SemanticFieldConstants.SEARCH_MODEL_ID;
import static org.opensearch.neuralsearch.constants.SemanticFieldConstants.SEMANTIC_INFO_FIELD_NAME;
import static org.opensearch.neuralsearch.constants.SemanticInfoFieldConstants.CHUNKS_EMBEDDING_FIELD_NAME;
import static org.opensearch.neuralsearch.constants.SemanticInfoFieldConstants.CHUNKS_FIELD_NAME;
import static org.opensearch.neuralsearch.processor.TextImageEmbeddingProcessor.EMBEDDING_FIELD;
import static org.opensearch.neuralsearch.processor.TextImageEmbeddingProcessor.INPUT_IMAGE;
import static org.opensearch.neuralsearch.processor.TextImageEmbeddingProcessor.INPUT_TEXT;
import static org.opensearch.neuralsearch.query.parser.NeuralQueryParser.indexToTargetFieldConfigStreamInput;
import static org.opensearch.neuralsearch.query.parser.NeuralQueryParser.indexToTargetFieldConfigStreamOutput;
import static org.opensearch.neuralsearch.query.parser.NeuralQueryParser.modelIdToQueryTokensSupplierMapStreamInput;
import static org.opensearch.neuralsearch.query.parser.NeuralQueryParser.modelIdToQueryTokensSupplierMapStreamOutput;
import static org.opensearch.neuralsearch.query.parser.NeuralQueryParser.modelIdToVectorSupplierMapStreamInput;
import static org.opensearch.neuralsearch.query.parser.NeuralQueryParser.modelIdToVectorSupplierMapStreamOutput;
import static org.opensearch.neuralsearch.query.parser.NeuralQueryParser.queryTokensMapSupplierStreamInput;
import static org.opensearch.neuralsearch.query.parser.NeuralQueryParser.queryTokensMapSupplierStreamOutput;
import static org.opensearch.neuralsearch.query.parser.NeuralQueryParser.vectorSupplierStreamInput;
import static org.opensearch.neuralsearch.query.parser.NeuralQueryParser.vectorSupplierStreamOutput;
import static org.opensearch.neuralsearch.util.NeuralQueryValidationUtil.validateNeuralQueryForKnn;
import static org.opensearch.neuralsearch.util.NeuralQueryValidationUtil.validateNeuralQueryForSemanticDense;
import static org.opensearch.neuralsearch.util.NeuralQueryValidationUtil.validateNeuralQueryForSemanticSparse;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import lombok.NonNull;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang.builder.EqualsBuilder;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.join.ScoreMode;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.SetOnce;
import org.opensearch.core.ParseField;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.ParsingException;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentLocation;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.NumberFieldMapper;
import org.opensearch.index.mapper.RankFeaturesFieldMapper;
import org.opensearch.index.query.AbstractQueryBuilder;
import org.opensearch.index.query.NestedQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryCoordinatorContext;
import org.opensearch.index.query.QueryRewriteContext;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.query.parser.MethodParametersParser;
import org.opensearch.knn.index.query.parser.RescoreParser;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.neuralsearch.common.MinClusterVersionUtil;
import org.opensearch.neuralsearch.mapper.SemanticFieldMapper;
import org.opensearch.neuralsearch.util.SemanticMappingUtils;
import org.opensearch.neuralsearch.ml.MLCommonsClientAccessor;

import com.google.common.annotations.VisibleForTesting;

import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;
import lombok.extern.log4j.Log4j2;
import org.opensearch.neuralsearch.processor.MapInferenceRequest;
import org.opensearch.neuralsearch.processor.TextInferenceRequest;
import org.opensearch.neuralsearch.query.dto.NeuralQueryTargetFieldConfig;
import org.opensearch.neuralsearch.util.FeatureFlagUtil;
import org.opensearch.neuralsearch.util.TokenWeightUtil;

/**
 * NeuralQueryBuilder is responsible for producing "neural" query types. A "neural" query type is a wrapper around a
 * k-NN vector query. It uses a ML language model to produce a dense vector from a query string that is then used as
 * the query vector for the k-NN search.
 */

@Log4j2
@Getter
@Setter
@Accessors(chain = true, fluent = true)
@NoArgsConstructor(access = AccessLevel.PRIVATE)
@AllArgsConstructor(access = AccessLevel.PRIVATE)
public class NeuralQueryBuilder extends AbstractQueryBuilder<NeuralQueryBuilder> implements ModelInferenceQueryBuilder {

    public static final String NAME = "neural";

    // common fields used for both dense and sparse model
    @VisibleForTesting
    public static final ParseField QUERY_TEXT_FIELD = new ParseField("query_text");

    public static final ParseField MODEL_ID_FIELD = new ParseField("model_id");

    // fields only used for dense model
    public static final ParseField QUERY_IMAGE_FIELD = new ParseField("query_image");

    @VisibleForTesting
    static final ParseField K_FIELD = new ParseField("k");

    public static final int DEFAULT_K = 10;

    // fields for sparse model
    public static final ParseField QUERY_TOKENS_FIELD = new ParseField("query_tokens");

    // client to invoke ml-common APIs
    private static MLCommonsClientAccessor ML_CLIENT;
    private static ClusterService CLUSTER_SERVICE;

    public static void initialize(MLCommonsClientAccessor mlClient, ClusterService clusterService) {
        NeuralQueryBuilder.ML_CLIENT = mlClient;
        NeuralQueryBuilder.CLUSTER_SERVICE = clusterService;
    }

    // common fields used for both dense and sparse model
    private String fieldName;
    private String queryText;
    private String modelId;
    private String embeddingFieldType;

    // fields only used for dense model
    private String queryImage;
    private Integer k = null;
    private Float maxDistance = null;
    private Float minScore = null;
    private Boolean expandNested;
    @VisibleForTesting
    @Getter(AccessLevel.PACKAGE)
    @Setter(AccessLevel.PACKAGE)
    private Supplier<float[]> vectorSupplier;
    private QueryBuilder filter;
    private Map<String, ?> methodParameters;
    private RescoreContext rescoreContext;
    // fields to support the semantic field for dense model
    private Map<String, Supplier<float[]>> modelIdToVectorSupplierMap;

    // fields only used for sparse model
    private Supplier<Map<String, Float>> queryTokensMapSupplier;
    // fields to support the semantic field for sparse model
    private Map<String, Supplier<Map<String, Float>>> modelIdToQueryTokensSupplierMap;

    private Map<String, NeuralQueryTargetFieldConfig> indexToTargetFieldConfig;

    /**
     * A custom builder class to enforce valid Neural Query Builder instantiation
     */
    public static class Builder {
        private String fieldName;
        private String queryText;
        private String queryImage;
        private String modelId;
        private Integer k = null;
        private Float maxDistance = null;
        private Float minScore = null;
        private Boolean expandNested;
        private Supplier<float[]> vectorSupplier;
        private QueryBuilder filter;
        private Map<String, ?> methodParameters;
        private RescoreContext rescoreContext;
        private String queryName;
        private float boost = DEFAULT_BOOST;
        private String embeddingFieldType;
        private Map<String, Supplier<float[]>> modelIdToVectorSupplierMap;
        private Supplier<Map<String, Float>> queryTokensMapSupplier;
        private Map<String, Supplier<Map<String, Float>>> modelIdToQueryTokensSupplierMap;
        private Map<String, NeuralQueryTargetFieldConfig> indexToTargetFieldConfig;

        public Builder() {}

        public Builder fieldName(String fieldName) {
            this.fieldName = fieldName;
            return this;
        }

        public Builder queryText(String queryText) {
            this.queryText = queryText;
            return this;
        }

        public Builder queryImage(String queryImage) {
            this.queryImage = queryImage;
            return this;
        }

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public Builder k(Integer k) {
            this.k = k;
            return this;
        }

        public Builder maxDistance(Float maxDistance) {
            this.maxDistance = maxDistance;
            return this;
        }

        public Builder minScore(Float minScore) {
            this.minScore = minScore;
            return this;
        }

        public Builder expandNested(Boolean expandNested) {
            this.expandNested = expandNested;
            return this;
        }

        public Builder vectorSupplier(Supplier<float[]> vectorSupplier) {
            this.vectorSupplier = vectorSupplier;
            return this;
        }

        public Builder filter(QueryBuilder filter) {
            this.filter = filter;
            return this;
        }

        public Builder methodParameters(Map<String, ?> methodParameters) {
            this.methodParameters = methodParameters;
            return this;
        }

        public Builder queryName(String queryName) {
            this.queryName = queryName;
            return this;
        }

        public Builder boost(float boost) {
            this.boost = boost;
            return this;
        }

        public Builder rescoreContext(RescoreContext rescoreContext) {
            this.rescoreContext = rescoreContext;
            return this;
        }

        public void embeddingFieldType(String embeddingFieldType) {
            this.embeddingFieldType = embeddingFieldType;
        }

        public void modelIdToVectorSupplierMap(Map<String, Supplier<float[]>> modelIdToVectorSupplierMap) {
            this.modelIdToVectorSupplierMap = modelIdToVectorSupplierMap;
        }

        public void queryTokensMapSupplier(Supplier<Map<String, Float>> queryTokensMapSupplier) {
            this.queryTokensMapSupplier = queryTokensMapSupplier;
        }

        public void modelIdToQueryTokensSupplierMap(Map<String, Supplier<Map<String, Float>>> modelIdToQueryTokensSupplierMap) {
            this.modelIdToQueryTokensSupplierMap = modelIdToQueryTokensSupplierMap;
        }

        public void indexToTargetFieldConfig(Map<String, NeuralQueryTargetFieldConfig> indexToTargetFieldConfig) {
            this.indexToTargetFieldConfig = indexToTargetFieldConfig;
        }

        public NeuralQueryBuilder build() {
            final NeuralQueryBuilder neuralQueryBuilder = new NeuralQueryBuilder(
                fieldName,
                queryText,
                modelId,
                embeddingFieldType,
                queryImage,
                k,
                maxDistance,
                minScore,
                expandNested,
                vectorSupplier,
                filter,
                methodParameters,
                rescoreContext,
                modelIdToVectorSupplierMap,
                queryTokensMapSupplier,
                modelIdToQueryTokensSupplierMap,
                indexToTargetFieldConfig
            ).boost(boost).queryName(queryName);

            List<String> errors;

            if (embeddingFieldType == null) {
                errors = validateNeuralQueryForKnn(neuralQueryBuilder);
            } else if (KNNVectorFieldMapper.CONTENT_TYPE.equals(embeddingFieldType)) {
                errors = validateNeuralQueryForSemanticDense(neuralQueryBuilder);
            } else if (RankFeaturesFieldMapper.CONTENT_TYPE.equals(embeddingFieldType)) {
                errors = validateNeuralQueryForSemanticSparse(neuralQueryBuilder);
            } else {
                throw new IllegalArgumentException("Unsupported embedding field type: " + embeddingFieldType);
            }
            if (!errors.isEmpty()) {
                throw new IllegalArgumentException("Failed to build the NeuralQueryBuilder: " + String.join("; ", errors));
            } else {
                return neuralQueryBuilder;
            }
        }

    }

    public static NeuralQueryBuilder.Builder builder() {
        return new NeuralQueryBuilder.Builder();
    }

    /**
     * Constructor from stream input
     *
     * @param in StreamInput to initialize object from
     * @throws IOException thrown if unable to read from input stream
     */
    public NeuralQueryBuilder(StreamInput in) throws IOException {
        super(in);
        this.fieldName = in.readString();
        // The query image field was introduced since v2.11.0 through the
        // https://github.com/opensearch-project/neural-search/pull/359 but at that time we didn't add it to
        // NeuralQueryBuilder(StreamInput in) and doWriteTo(StreamOutput out) function. The fix will be
        // introduced in v2.19.0 so we need this check for the backward compatibility.
        if (isClusterOnOrAfterMinReqVersion(QUERY_IMAGE_FIELD.getPreferredName())) {
            this.queryText = in.readOptionalString();
            this.queryImage = in.readOptionalString();
        } else {
            this.queryText = in.readString();
        }
        // If cluster version is on or after 2.11 then default model Id support is enabled
        if (isClusterOnOrAfterMinReqVersionForDefaultDenseModelIdSupport()) {
            this.modelId = in.readOptionalString();
        } else {
            this.modelId = in.readString();
        }
        if (isClusterOnOrAfterMinReqVersionForRadialSearch()) {
            this.k = in.readOptionalInt();
        } else {
            this.k = in.readVInt();
        }
        this.filter = in.readOptionalNamedWriteable(QueryBuilder.class);
        if (isClusterOnOrAfterMinReqVersionForRadialSearch()) {
            this.maxDistance = in.readOptionalFloat();
            this.minScore = in.readOptionalFloat();
        }
        if (isClusterOnOrAfterMinReqVersion(EXPAND_NESTED_FIELD.getPreferredName())) {
            this.expandNested = in.readOptionalBoolean();
        }
        if (isClusterOnOrAfterMinReqVersion(METHOD_PARAMS_FIELD.getPreferredName())) {
            this.methodParameters = MethodParametersParser.streamInput(in, MinClusterVersionUtil::isClusterOnOrAfterMinReqVersion);
        }
        this.rescoreContext = RescoreParser.streamInput(in);
        if (FeatureFlagUtil.isEnabled(FeatureFlagUtil.SEMANTIC_FIELD_ENABLED)) {
            this.vectorSupplier = vectorSupplierStreamInput(in);
            this.modelIdToVectorSupplierMap = modelIdToVectorSupplierMapStreamInput(in);
            this.queryTokensMapSupplier = queryTokensMapSupplierStreamInput(in);
            this.modelIdToQueryTokensSupplierMap = modelIdToQueryTokensSupplierMapStreamInput(in);
            this.indexToTargetFieldConfig = indexToTargetFieldConfigStreamInput(in);
        }
    }

    @Override
    protected void doWriteTo(StreamOutput out) throws IOException {
        out.writeString(this.fieldName);
        // The query image field was introduced since v2.11.0 through the
        // https://github.com/opensearch-project/neural-search/pull/359 but at that time we didn't add it to
        // NeuralQueryBuilder(StreamInput in) and doWriteTo(StreamOutput out) function. The fix will be
        // introduced in v2.19.0 so we need this check for the backward compatibility.
        if (isClusterOnOrAfterMinReqVersion(QUERY_IMAGE_FIELD.getPreferredName())) {
            out.writeOptionalString(this.queryText);
            out.writeOptionalString(this.queryImage);
        } else {
            out.writeString(this.queryText);
        }
        // If cluster version is on or after 2.11 then default model Id support is enabled
        if (isClusterOnOrAfterMinReqVersionForDefaultDenseModelIdSupport()) {
            out.writeOptionalString(this.modelId);
        } else {
            out.writeString(this.modelId);
        }
        if (isClusterOnOrAfterMinReqVersionForRadialSearch()) {
            out.writeOptionalInt(this.k);
        } else {
            out.writeVInt(this.k);
        }
        out.writeOptionalNamedWriteable(this.filter);
        if (isClusterOnOrAfterMinReqVersionForRadialSearch()) {
            out.writeOptionalFloat(this.maxDistance);
            out.writeOptionalFloat(this.minScore);
        }
        if (isClusterOnOrAfterMinReqVersion(EXPAND_NESTED_FIELD.getPreferredName())) {
            out.writeOptionalBoolean(this.expandNested);
        }

        if (isClusterOnOrAfterMinReqVersion(METHOD_PARAMS_FIELD.getPreferredName())) {
            MethodParametersParser.streamOutput(out, methodParameters, MinClusterVersionUtil::isClusterOnOrAfterMinReqVersion);
        }
        RescoreParser.streamOutput(out, rescoreContext);

        if (FeatureFlagUtil.isEnabled(FeatureFlagUtil.SEMANTIC_FIELD_ENABLED)) {
            vectorSupplierStreamOutput(out, vectorSupplier);
            modelIdToVectorSupplierMapStreamOutput(out, modelIdToVectorSupplierMap);
            queryTokensMapSupplierStreamOutput(out, queryTokensMapSupplier);
            modelIdToQueryTokensSupplierMapStreamOutput(out, modelIdToQueryTokensSupplierMap);
            indexToTargetFieldConfigStreamOutput(out, indexToTargetFieldConfig);
        }
    }

    @Override
    protected void doXContent(XContentBuilder xContentBuilder, Params params) throws IOException {
        xContentBuilder.startObject(NAME);
        xContentBuilder.startObject(fieldName);
        if (Objects.nonNull(queryText)) {
            xContentBuilder.field(QUERY_TEXT_FIELD.getPreferredName(), queryText);
        }
        if (Objects.nonNull(queryImage)) {
            xContentBuilder.field(QUERY_IMAGE_FIELD.getPreferredName(), queryImage);
        }
        if (Objects.nonNull(modelId)) {
            xContentBuilder.field(MODEL_ID_FIELD.getPreferredName(), modelId);
        }
        if (Objects.nonNull(k)) {
            xContentBuilder.field(K_FIELD.getPreferredName(), k);
        }
        if (Objects.nonNull(filter)) {
            xContentBuilder.field(FILTER_FIELD.getPreferredName(), filter);
        }
        if (Objects.nonNull(maxDistance)) {
            xContentBuilder.field(MAX_DISTANCE_FIELD.getPreferredName(), maxDistance);
        }
        if (Objects.nonNull(minScore)) {
            xContentBuilder.field(MIN_SCORE_FIELD.getPreferredName(), minScore);
        }
        if (Objects.nonNull(expandNested)) {
            xContentBuilder.field(EXPAND_NESTED_FIELD.getPreferredName(), expandNested);
        }
        if (Objects.nonNull(methodParameters)) {
            MethodParametersParser.doXContent(xContentBuilder, methodParameters);
        }
        if (Objects.nonNull(rescoreContext)) {
            RescoreParser.doXContent(xContentBuilder, rescoreContext);
        }
        if (Objects.nonNull(queryTokensMapSupplier) && Objects.nonNull(queryTokensMapSupplier.get())) {
            xContentBuilder.field(QUERY_TOKENS_FIELD.getPreferredName(), queryTokensMapSupplier.get());
        }
        printBoostAndQueryName(xContentBuilder);
        xContentBuilder.endObject();
        xContentBuilder.endObject();
    }

    /**
     * Creates NeuralQueryBuilder from xContent.
     *
     * The expected parsing form looks like:
     * {
     *  "VECTOR_FIELD": {
     *    "query_text": "string",
     *    "model_id": "string",
     *    "k": int,
     *    "name": "string", (optional)
     *    "boost": float (optional),
     *    "filter": map (optional)
     *  }
     * }
     *
     * @param parser XContentParser
     * @return NeuralQueryBuilder
     * @throws IOException can be thrown by parser
     */
    public static NeuralQueryBuilder fromXContent(XContentParser parser) throws IOException {
        NeuralQueryBuilder neuralQueryBuilder = new NeuralQueryBuilder();
        if (parser.currentToken() != XContentParser.Token.START_OBJECT) {
            throw new ParsingException(parser.getTokenLocation(), "Token must be START_OBJECT");
        }
        parser.nextToken();
        neuralQueryBuilder.fieldName(parser.currentName());
        parser.nextToken();
        parseQueryParams(parser, neuralQueryBuilder);
        if (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            throw new ParsingException(
                parser.getTokenLocation(),
                "["
                    + NAME
                    + "] query doesn't support multiple fields, found ["
                    + neuralQueryBuilder.fieldName()
                    + "] and ["
                    + parser.currentName()
                    + "]"
            );
        }
        requireValue(neuralQueryBuilder.fieldName, "Field name must be provided for neural query");

        if (!FeatureFlagUtil.isEnabled(FeatureFlagUtil.SEMANTIC_FIELD_ENABLED)) {
            // To support semantic field we should delay this validation until we pull more detail from the index
            // mapping during the query rewrite
            if (StringUtils.isBlank(neuralQueryBuilder.queryText()) && StringUtils.isBlank(neuralQueryBuilder.queryImage())) {
                throw new IllegalArgumentException("Either query text or image text must be provided for neural query");
            }
            if (!isClusterOnOrAfterMinReqVersionForDefaultDenseModelIdSupport()) {
                requireValue(neuralQueryBuilder.modelId(), "Model ID must be provided for neural query");
            }
            boolean queryTypeIsProvided = validateKNNQueryType(
                neuralQueryBuilder.k(),
                neuralQueryBuilder.maxDistance(),
                neuralQueryBuilder.minScore()
            );
            if (queryTypeIsProvided == false) {
                neuralQueryBuilder.k(DEFAULT_K);
            }
        }
        return neuralQueryBuilder;
    }

    private static void parseQueryParams(XContentParser parser, NeuralQueryBuilder neuralQueryBuilder) throws IOException {
        XContentParser.Token token;
        String currentFieldName = "";
        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                currentFieldName = parser.currentName();
            } else if (token.isValue()) {
                if (QUERY_TEXT_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                    neuralQueryBuilder.queryText(parser.text());
                } else if (QUERY_IMAGE_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                    neuralQueryBuilder.queryImage(parser.text());
                } else if (MODEL_ID_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                    neuralQueryBuilder.modelId(parser.text());
                } else if (K_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                    neuralQueryBuilder.k((Integer) NumberFieldMapper.NumberType.INTEGER.parse(parser.objectBytes(), false));
                } else if (NAME_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                    neuralQueryBuilder.queryName(parser.text());
                } else if (BOOST_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                    neuralQueryBuilder.boost(parser.floatValue());
                } else if (MAX_DISTANCE_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                    neuralQueryBuilder.maxDistance(parser.floatValue());
                } else if (MIN_SCORE_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                    neuralQueryBuilder.minScore(parser.floatValue());
                } else if (EXPAND_NESTED_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                    neuralQueryBuilder.expandNested(parser.booleanValue());
                } else {
                    throw getUnsupportedFieldException(parser.getTokenLocation(), currentFieldName);
                }
            } else if (token == XContentParser.Token.START_OBJECT) {
                if (FILTER_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                    neuralQueryBuilder.filter(parseInnerQueryBuilder(parser));
                } else if (METHOD_PARAMS_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                    neuralQueryBuilder.methodParameters(MethodParametersParser.fromXContent(parser));
                } else if (RESCORE_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                    neuralQueryBuilder.rescoreContext(RescoreParser.fromXContent(parser));
                } else if (QUERY_TOKENS_FIELD.match(currentFieldName, parser.getDeprecationHandler())) {
                    if (FeatureFlagUtil.isEnabled(FeatureFlagUtil.SEMANTIC_FIELD_ENABLED)) {
                        final Map<String, Float> queryTokens = parser.map(HashMap::new, XContentParser::floatValue);
                        neuralQueryBuilder.queryTokensMapSupplier(() -> queryTokens);
                    } else {
                        throw getUnsupportedFieldException(parser.getTokenLocation(), currentFieldName);
                    }
                } else {
                    throw getUnsupportedFieldException(parser.getTokenLocation(), currentFieldName);
                }
            } else {
                throw new ParsingException(
                    parser.getTokenLocation(),
                    "[" + NAME + "] unknown token [" + token + "] after [" + currentFieldName + "]"
                );
            }
        }
    }

    private static ParsingException getUnsupportedFieldException(XContentLocation tokenLocation, String currentFieldName) {
        return new ParsingException(tokenLocation, "[" + NAME + "] query does not support [" + currentFieldName + "]");
    }

    @Override
    protected QueryBuilder doRewrite(QueryRewriteContext queryRewriteContext) {
        final QueryCoordinatorContext coordinatorContext = queryRewriteContext.convertToCoordinatorContext();
        if (coordinatorContext != null) {
            prepareRewriteForTargetIndices(coordinatorContext);
        }

        final NeuralQueryTargetFieldConfig config = getFirstTargetFieldConfig();

        // If no target field config found or the target field is not a semantic field then fall back to the old logic.
        // Here we just need to check the target field config in one index since we already validate the target field
        // in all the target indices should all be semantic field or non-semantic field.
        if (config == null || Boolean.FALSE.equals(config.getIsSemanticField())) {
            // When the target field is not a semantic field we simply fall back to the old logic to support the
            // dense model. But in the future we can add the logic to also support the sparse model.
            return rewriteQueryAgainstKnnField(queryRewriteContext);
        } else {
            return rewriteQueryAgainstSemanticField(queryRewriteContext);
        }
    }

    private NeuralQueryTargetFieldConfig getFirstTargetFieldConfig() {
        final Set<String> targetIndices = indexToTargetFieldConfig.keySet();
        if (targetIndices.isEmpty()) {
            return null;
        }
        return indexToTargetFieldConfig.get(targetIndices.iterator().next());
    }

    private void prepareRewriteForTargetIndices(final @NonNull QueryCoordinatorContext coordinatorContext) {
        // Skip if index-to-target-field config already exists—this can happen after async inference or when the
        // query is forwarded to another node.
        if (indexToTargetFieldConfig != null) {
            return;
        }

        indexToTargetFieldConfig = new HashMap<>();
        final List<IndexMetadata> targetIndexMetadataList = coordinatorContext.getTargetIndexMetadataList();
        if (targetIndexMetadataList == null) {
            return;
        }

        extractTargetFieldConfig(targetIndexMetadataList);

        validateTargetFieldConfig();
    }

    private void extractTargetFieldConfig(@NonNull final List<IndexMetadata> targetIndexMetadataList) {
        for (IndexMetadata indexMetadata : targetIndexMetadataList) {
            final MappingMetadata mappingMetadata = indexMetadata.mapping();
            final NeuralQueryTargetFieldConfig.NeuralQueryTargetFieldConfigBuilder targetFieldConfigBuilder = NeuralQueryTargetFieldConfig
                .builder();
            if (mappingMetadata == null) {
                indexToTargetFieldConfig.put(indexMetadata.getIndex().toString(), targetFieldConfigBuilder.isUnmappedField(true).build());
                continue;
            }
            final Map<String, Object> mappings = mappingMetadata.sourceAsMap();
            final Map<String, Object> targetFieldConfig = SemanticMappingUtils.getFieldConfigByPath(mappings, fieldName);
            if (targetFieldConfig == null) {
                indexToTargetFieldConfig.put(indexMetadata.getIndex().toString(), targetFieldConfigBuilder.isUnmappedField(true).build());
                continue;
            }
            targetFieldConfigBuilder.isUnmappedField(false);
            final Object targetFieldTypeObject = targetFieldConfig.get(TYPE);
            if (!(targetFieldTypeObject instanceof String targetFieldType)) {
                throw new IllegalArgumentException(
                    "Failed to process the neural query against the field [" + fieldName + "]" + " because it is an object field."
                );
            }
            final boolean isSemanticField = SemanticFieldMapper.CONTENT_TYPE.equals(targetFieldType);
            if (isSemanticField) {
                final Map<String, Object> embeddingFieldConfig = getSemanticEmbeddingFieldConfig(targetFieldConfig, mappings);
                final String embeddingFieldType = (String) embeddingFieldConfig.get(TYPE);
                String searchModelId = (String) targetFieldConfig.get(MODEL_ID);
                if (targetFieldConfig.containsKey(SEARCH_MODEL_ID)) {
                    searchModelId = (String) targetFieldConfig.get(SEARCH_MODEL_ID);
                }
                targetFieldConfigBuilder.embeddingFieldType(embeddingFieldType);
                targetFieldConfigBuilder.searchModelId(searchModelId);
                targetFieldConfigBuilder.isSemanticField(Boolean.TRUE);
            } else {
                targetFieldConfigBuilder.isSemanticField(Boolean.FALSE);
            }
            indexToTargetFieldConfig.put(indexMetadata.getIndex().toString(), targetFieldConfigBuilder.build());
        }
    }

    private Map<String, Object> getSemanticEmbeddingFieldConfig(
        @NonNull final Map<String, Object> targetFieldConfig,
        @NonNull final Map<String, Object> mappings
    ) {
        String embeddingFieldPath = fieldName + DEFAULT_SEMANTIC_INFO_FIELD_NAME_SUFFIX;
        if (targetFieldConfig.containsKey(SEMANTIC_INFO_FIELD_NAME)) {
            String[] paths = embeddingFieldPath.split("\\.");
            paths[paths.length - 1] = (String) targetFieldConfig.get(SEMANTIC_INFO_FIELD_NAME);
            embeddingFieldPath = String.join(".", paths);
        }
        embeddingFieldPath += PATH_SEPARATOR + CHUNKS_FIELD_NAME + PATH_SEPARATOR + EMBEDDING_FIELD;
        return SemanticMappingUtils.getFieldConfigByPath(mappings, embeddingFieldPath);
    }

    private void validateTargetFieldConfig() {
        final List<String> indicesWithSemantic = new ArrayList<>();
        final List<String> indicesWithNonSemantic = new ArrayList<>();
        final List<String> indicesWithSemanticDense = new ArrayList<>();
        final List<String> indicesWithSemanticSparse = new ArrayList<>();
        List<String> validationErrors = new ArrayList<>();

        for (Map.Entry<String, NeuralQueryTargetFieldConfig> entry : indexToTargetFieldConfig.entrySet()) {
            final String targetIndex = entry.getKey();
            final NeuralQueryTargetFieldConfig targetFieldConfig = entry.getValue();
            if (!targetFieldConfig.getIsUnmappedField()) {
                if (targetFieldConfig.getIsSemanticField()) {
                    indicesWithSemantic.add(targetIndex);
                    switch (targetFieldConfig.getEmbeddingFieldType()) {
                        case KNNVectorFieldMapper.CONTENT_TYPE -> indicesWithSemanticDense.add(targetIndex);
                        case RankFeaturesFieldMapper.CONTENT_TYPE -> indicesWithSemanticSparse.add(targetIndex);
                        default -> validationErrors.add(
                            "Unsupported embedding field type ["
                                + targetFieldConfig.getEmbeddingFieldType()
                                + "] in the target index ["
                                + targetIndex
                                + "]"
                        );
                    }
                } else {
                    indicesWithNonSemantic.add(targetIndex);
                }
            }
            // If the target field in the target index is an unmapped field we don't process it here.
            // Later in the doToQuery function we will convert it to MatchNoDocsQuery.
        }

        if (!indicesWithSemantic.isEmpty() && !indicesWithNonSemantic.isEmpty()) {
            validationErrors.add(
                "The target field should be either a semantic field or a non-semantic field in all "
                    + "the target indices. It is a semantic field in indices: "
                    + String.join(", ", indicesWithSemantic)
                    + " while not a semantic field in indices "
                    + String.join(", ", indicesWithNonSemantic)
            );
        } else if (!indicesWithSemantic.isEmpty()) {
            if (!indicesWithSemanticDense.isEmpty() && !indicesWithSemanticSparse.isEmpty()) {
                validationErrors.add(
                    "The target semantic field should be either use a dense model or a sparse model"
                        + " in all the target indices. It is a dense model in indices: "
                        + String.join(", ", indicesWithSemanticDense)
                        + " while a sparse model in indices "
                        + String.join(", ", indicesWithSemanticSparse)
                );

            } else if (!indicesWithSemanticDense.isEmpty()) {
                validateNeuralQueryForSemanticDense(this);
            } else if (!indicesWithSemanticSparse.isEmpty()) {
                validateNeuralQueryForSemanticSparse(this);
            }
        } else if (!indicesWithNonSemantic.isEmpty()) {
            validationErrors.addAll(validateNeuralQueryForKnn(this));
        }

        if (!validationErrors.isEmpty()) {
            throw new IllegalArgumentException(
                "Invalid neural query against field " + fieldName + ". Errors: " + String.join("; ", validationErrors)
            );
        }
    }

    private QueryBuilder rewriteQueryAgainstKnnField(QueryRewriteContext queryRewriteContext) {
        // When re-writing a QueryBuilder, if the QueryBuilder is not changed, doRewrite should return itself
        // (see
        // https://github.com/opensearch-project/OpenSearch/blob/main/server/src/main/java/org/opensearch/index/query/QueryBuilder.java#L90-L98).
        // Otherwise, it should return the modified copy (see rewrite logic
        // https://github.com/opensearch-project/OpenSearch/blob/main/server/src/main/java/org/opensearch/index/query/Rewriteable.java#L117.
        // With the asynchronous call, on first rewrite, we create a new
        // vector supplier that will get populated once the asynchronous call finishes and pass this supplier in to
        // create a new builder. Once the supplier's value gets set, we return a NeuralKNNQueryBuilder
        // which wrapped KNNQueryBuilder. Otherwise, we just return the current unmodified query builder.
        if (vectorSupplier() != null) {
            if (vectorSupplier().get() == null) {
                return this;
            }

            return NeuralKNNQueryBuilder.builder()
                .fieldName(fieldName())
                .vector(vectorSupplier.get())
                .k(k())
                .filter(filter())
                .maxDistance(maxDistance())
                .minScore(minScore())
                .expandNested(expandNested())
                .methodParameters(methodParameters())
                .rescoreContext(rescoreContext())
                .originalQueryText(queryText())
                .build();
        }

        SetOnce<float[]> vectorSetOnce = new SetOnce<>();
        Map<String, String> inferenceInput = new HashMap<>();
        if (StringUtils.isNotBlank(queryText())) {
            inferenceInput.put(INPUT_TEXT, queryText());
        }
        if (StringUtils.isNotBlank(queryImage())) {
            inferenceInput.put(INPUT_IMAGE, queryImage());
        }
        queryRewriteContext.registerAsyncAction(
            ((client, actionListener) -> ML_CLIENT.inferenceSentencesMap(
                MapInferenceRequest.builder().modelId(modelId()).inputObjects(inferenceInput).build(),
                ActionListener.wrap(floatList -> {
                    vectorSetOnce.set(vectorAsListToArray(floatList));
                    actionListener.onResponse(null);
                }, actionListener::onFailure)
            ))
        );
        return new NeuralQueryBuilder(
            fieldName(),
            queryText(),
            modelId(),
            KNNVectorFieldMapper.CONTENT_TYPE,
            queryImage(),
            k(),
            maxDistance(),
            minScore(),
            expandNested(),
            vectorSetOnce::get,
            filter(),
            methodParameters(),
            rescoreContext(),
            null,
            null,
            null,
            null
        );
    }

    private QueryBuilder rewriteQueryAgainstSemanticField(@NonNull final QueryRewriteContext queryRewriteContext) {
        final QueryShardContext shardContext = queryRewriteContext.convertToShardContext();
        final QueryCoordinatorContext coordinatorContext = queryRewriteContext.convertToCoordinatorContext();

        if (coordinatorContext != null) {
            return inference(coordinatorContext);
        } else if (shardContext != null) {
            return rewriteQueryAgainstSemanticFieldOnShard(shardContext);
        } else {
            // why we do rewrite here?
            return this;
        }
    }

    private QueryBuilder rewriteQueryAgainstSemanticFieldOnShard(QueryShardContext shardContext) {
        final MappedFieldType mappedFieldType = shardContext.fieldMapper(fieldName());
        if (mappedFieldType == null) {
            // We will convert it to NoMatchDocQuery later in doToQuery function.
            return this;
        }
        if (SemanticFieldMapper.CONTENT_TYPE.equals(mappedFieldType.typeName())) {
            final SemanticFieldMapper.SemanticFieldType semanticFieldType = (SemanticFieldMapper.SemanticFieldType) mappedFieldType;
            final String searchModelId = getSearchModelId(semanticFieldType);
            assert searchModelId != null : "Search model id must exist.";

            final String nestedQueryPath = getNestedQueryPath(semanticFieldType);
            final String embeddingFieldName = nestedQueryPath + PATH_SEPARATOR + CHUNKS_EMBEDDING_FIELD_NAME;

            final MappedFieldType embeddingFieldType = shardContext.fieldMapper(embeddingFieldName);
            if (embeddingFieldType == null) {
                throw new RuntimeException(
                    getErrorMessageWithBaseError("Expect the embedding field exists in the index mapping but not able to find it.")
                );
            }
            if (KNNVectorFieldMapper.CONTENT_TYPE.equals(embeddingFieldType.unwrap().typeName())) {
                float[] vector;
                try {
                    vector = modelIdToVectorSupplierMap.get(searchModelId).get();
                } catch (Exception e) {
                    throw new RuntimeException(
                        getErrorMessageWithBaseError("Not able to find the dense embedding when try to rewrite it on the shard level."),
                        e
                    );
                }
                final KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
                    .fieldName(embeddingFieldName)
                    .vector(vector)
                    .filter(filter())
                    .maxDistance(maxDistance)
                    .minScore(minScore)
                    .expandNested(expandNested)
                    .k(k)
                    .methodParameters(methodParameters)
                    .rescoreContext(rescoreContext)
                    .build();
                return new NestedQueryBuilder(nestedQueryPath, knnQueryBuilder, ScoreMode.Max);
            } else if (RankFeaturesFieldMapper.CONTENT_TYPE.equals(embeddingFieldType.unwrap().typeName())) {
                Supplier<Map<String, Float>> queryTokensSupplier;
                try {
                    queryTokensSupplier = modelIdToQueryTokensSupplierMap.get(searchModelId);
                } catch (Exception e) {
                    throw new RuntimeException(
                        getErrorMessageWithBaseError("Not able to find the sparse embedding when try to rewrite it on the shard level."),
                        e
                    );
                }
                final NeuralSparseQueryBuilder neuralSparseQueryBuilder = new NeuralSparseQueryBuilder().fieldName(embeddingFieldName)
                    .queryTokensSupplier(queryTokensSupplier);
                return new NestedQueryBuilder(nestedQueryPath, neuralSparseQueryBuilder, ScoreMode.Max);
            } else {
                throw new RuntimeException(
                    getErrorMessageWithBaseError(
                        "Expect the embedding field type to be knn_vector or ran_features but found unsupported embedding field type: "
                            + embeddingFieldType.unwrap().typeName()
                    )
                );
            }
        } else {
            throw new RuntimeException("Expect the neural query target field is a semantic field but found: " + mappedFieldType.typeName());
        }
    }

    private String getSearchModelId(@NonNull final SemanticFieldMapper.SemanticFieldType semanticFieldType) {
        if (modelId != null) {
            return modelId;
        } else if (semanticFieldType.getSemanticParameters().getSearchModelId() != null) {
            return semanticFieldType.getSemanticParameters().getSearchModelId();
        } else {
            return semanticFieldType.getSemanticParameters().getModelId();
        }
    }

    private String getErrorMessageWithBaseError(@NonNull final String errorMessage) {
        return "Failed to execute the neural query against the semantic field " + fieldName + ". " + errorMessage;
    }

    private String getNestedQueryPath(SemanticFieldMapper.SemanticFieldType semanticTextFieldType) {
        final String[] paths = semanticTextFieldType.name().split("\\.");
        final String semanticInfoFieldName = semanticTextFieldType.getSemanticParameters().getSemanticInfoFieldName();
        paths[paths.length - 1] = semanticInfoFieldName == null
            ? paths[paths.length - 1] + DEFAULT_SEMANTIC_INFO_FIELD_NAME_SUFFIX
            : semanticInfoFieldName;
        return String.join(PATH_SEPARATOR, paths) + PATH_SEPARATOR + CHUNKS_FIELD_NAME;
    }

    private QueryBuilder inference(@NonNull final QueryCoordinatorContext queryRewriteContext) {
        // If it is not null it means we already start the async actions in previous rewrite.
        // Current rewrite happens after all the async actions done so simply return this to end the rewrite.
        if (modelIdToVectorSupplierMap != null || modelIdToQueryTokensSupplierMap != null) {
            return this;
        }

        Set<String> modelIds;
        if (modelId != null) {
            // If user explicitly define a model id in the query we should use it to override
            // the model id defined in the index mapping.
            modelIds = Set.of(modelId);
        } else {
            modelIds = indexToTargetFieldConfig.values()
                .stream()
                .map(NeuralQueryTargetFieldConfig::getSearchModelId)
                .collect(Collectors.toSet());
        }

        final NeuralQueryTargetFieldConfig config = getFirstTargetFieldConfig();
        assert config != null : "We should run the old query logic if no target field config found.";
        if (KNNVectorFieldMapper.CONTENT_TYPE.equals(config.getEmbeddingFieldType())) {
            inferenceByDenseModel(modelIds, queryRewriteContext);
        } else if (RankFeaturesFieldMapper.CONTENT_TYPE.equals(config.getEmbeddingFieldType())) {
            inferenceBySparseModel(modelIds, queryRewriteContext);
        } else {
            throw new RuntimeException(
                "Not able to do inference for the neural query against field "
                    + fieldName
                    + ". Unsupported embedding field type: "
                    + config.getEmbeddingFieldType()
            );
        }

        // We don't do rewrite just start the async actions to inference the query text
        // We still need to return a different object to enter the code block to execute the async tasks
        // Otherwise we will directly end the rewrite.
        return new NeuralQueryBuilder(
            fieldName(),
            queryText(),
            modelId(),
            config.getEmbeddingFieldType(),
            queryImage(),
            k(),
            maxDistance(),
            minScore(),
            expandNested(),
            vectorSupplier(),
            filter(),
            methodParameters(),
            rescoreContext(),
            modelIdToVectorSupplierMap(),
            queryTokensMapSupplier(),
            modelIdToQueryTokensSupplierMap(),
            indexToTargetFieldConfig()
        );

    }

    private void inferenceBySparseModel(@NonNull final Set<String> modelIds, @NonNull QueryCoordinatorContext queryRewriteContext) {
        modelIdToQueryTokensSupplierMap = new HashMap<>(modelIds.size());
        for (String modelId : modelIds) {
            final SetOnce<Map<String, Float>> setOnce = new SetOnce<>();
            modelIdToQueryTokensSupplierMap.put(modelId, setOnce::get);
            queryRewriteContext.registerAsyncAction(
                ((client, actionListener) -> ML_CLIENT.inferenceSentencesWithMapResult(
                    TextInferenceRequest.builder().modelId(modelId).inputTexts(List.of(queryText)).build(),
                    ActionListener.wrap(mapResultList -> {
                        final Map<String, Float> queryTokens = TokenWeightUtil.fetchListOfTokenWeightMap(mapResultList).get(0);
                        // Currently we don't support NeuralSparseTwoPhaseProcessor which can be supported
                        // in the future.
                        setOnce.set(queryTokens);
                        actionListener.onResponse(null);
                    }, actionListener::onFailure)
                ))
            );
        }
    }

    private void inferenceByDenseModel(@NonNull final Set<String> modelIds, @NonNull QueryCoordinatorContext queryRewriteContext) {
        final Map<String, String> inferenceInput = getInferenceInputForDenseModel();
        modelIdToVectorSupplierMap = new HashMap<>(modelIds.size());
        for (String modelId : modelIds) {
            final SetOnce<float[]> vectorSetOnce = new SetOnce<>();
            modelIdToVectorSupplierMap.put(modelId, vectorSetOnce::get);
            queryRewriteContext.registerAsyncAction(
                ((client, actionListener) -> ML_CLIENT.inferenceSentencesMap(
                    MapInferenceRequest.builder().modelId(modelId).inputObjects(inferenceInput).build(),
                    ActionListener.wrap(floatList -> {
                        vectorSetOnce.set(vectorAsListToArray(floatList));
                        actionListener.onResponse(null);
                    }, actionListener::onFailure)
                ))
            );
        }
    }

    private Map<String, String> getInferenceInputForDenseModel() {
        Map<String, String> inferenceInput = new HashMap<>();
        if (StringUtils.isNotBlank(queryText())) {
            inferenceInput.put(INPUT_TEXT, queryText());
        }
        if (StringUtils.isNotBlank(queryImage())) {
            inferenceInput.put(INPUT_IMAGE, queryImage());
        }
        return inferenceInput;
    }

    /**
     * We only rely on this function to handle the case when the target field is an unmapped field and simply convert
     * it to a MatchNoDocsQuery. For other use cases the query should be rewritten to other query builders, and we
     * should not reach this function.
     * @param queryShardContext query context on shard level
     * @return query
     */
    @Override
    protected Query doToQuery(QueryShardContext queryShardContext) {
        final MappedFieldType mappedFieldType = queryShardContext.fieldMapper(this.fieldName);
        if (mappedFieldType == null) {
            return new MatchNoDocsQuery();
        } else {
            throw new UnsupportedOperationException("Query cannot be created by NeuralQueryBuilder directly");
        }
    }

    @Override
    protected boolean doEquals(NeuralQueryBuilder obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        EqualsBuilder equalsBuilder = new EqualsBuilder();
        equalsBuilder.append(fieldName, obj.fieldName);
        equalsBuilder.append(queryText, obj.queryText);
        equalsBuilder.append(queryImage, obj.queryImage);
        equalsBuilder.append(modelId, obj.modelId);
        equalsBuilder.append(k, obj.k);
        equalsBuilder.append(maxDistance, obj.maxDistance);
        equalsBuilder.append(minScore, obj.minScore);
        equalsBuilder.append(expandNested, obj.expandNested);
        equalsBuilder.append(getVector(vectorSupplier), getVector(obj.vectorSupplier));
        equalsBuilder.append(filter, obj.filter);
        equalsBuilder.append(methodParameters, obj.methodParameters);
        equalsBuilder.append(rescoreContext, obj.rescoreContext);
        equalsBuilder.append(getQueryTokenMap(queryTokensMapSupplier), getQueryTokenMap(obj.queryTokensMapSupplier));
        return equalsBuilder.isEquals();
    }

    @Override
    protected int doHashCode() {
        return Objects.hash(
            fieldName,
            queryText,
            queryImage,
            modelId,
            k,
            maxDistance,
            minScore,
            expandNested,
            Arrays.hashCode(getVector(vectorSupplier)),
            filter,
            methodParameters,
            rescoreContext,
            getQueryTokenMap(queryTokensMapSupplier)
        );
    }

    private float[] getVector(final Supplier<float[]> vectorSupplier) {
        return Objects.isNull(vectorSupplier) ? null : vectorSupplier.get();
    }

    private Map<String, Float> getQueryTokenMap(final Supplier<Map<String, Float>> queryTokensSupplierMap) {
        return Objects.isNull(queryTokensSupplierMap) ? null : queryTokensSupplierMap.get();
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }

    private static boolean validateKNNQueryType(Integer k, Float maxDistance, Float minScore) {
        int queryCount = 0;
        if (k != null) {
            queryCount++;
        }
        if (maxDistance != null) {
            queryCount++;
        }
        if (minScore != null) {
            queryCount++;
        }
        if (queryCount > 1) {
            throw new IllegalArgumentException("Only one of k, max_distance, or min_score can be provided");
        }
        return queryCount == 1;
    }
}
