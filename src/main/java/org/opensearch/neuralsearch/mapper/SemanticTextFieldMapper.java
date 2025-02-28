/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.mapper;

import lombok.Getter;
import lombok.Setter;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.IndexOptions;
import org.opensearch.Version;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.analysis.IndexAnalyzers;
import org.opensearch.index.analysis.NamedAnalyzer;
import org.opensearch.index.mapper.ParametrizedFieldMapper;
import org.opensearch.index.mapper.TextFieldMapper;
import org.opensearch.index.mapper.TextParams;
import org.opensearch.index.mapper.TextSearchInfo;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.opensearch.neuralsearch.common.SemanticFieldConstants.MODEL_ID;
import static org.opensearch.neuralsearch.common.SemanticFieldConstants.RAW_FIELD_TYPE;
import static org.opensearch.neuralsearch.common.SemanticFieldConstants.SEMANTIC_INFO_FIELD_NAME;
import static org.opensearch.neuralsearch.common.SemanticFieldConstants.SEMANTIC_INFO_GENERATION_MODE;

/**
 * Defining how a semantic_text field is stored, indexed, and queried.
 */
public class SemanticTextFieldMapper extends TextFieldMapper {
    protected String modelId;
    protected String rawFieldType;
    protected String semanticInfoFieldName;
    protected String semanticInfoGenerationMode;

    protected SemanticTextFieldMapper(
        String simpleName,
        FieldType fieldType,
        TextFieldType mappedFieldType,
        PrefixFieldMapper prefixFieldMapper,
        PhraseFieldMapper phraseFieldMapper,
        MultiFields multiFields,
        CopyTo copyTo,
        SemanticTextFieldMapper.Builder builder
    ) {
        super(simpleName, fieldType, mappedFieldType, prefixFieldMapper, phraseFieldMapper, multiFields, copyTo, builder);
        this.modelId = builder.modelId.getValue();
        this.rawFieldType = builder.rawFieldType.getValue();
        this.semanticInfoFieldName = builder.semanticInfoFieldName.getValue();
        this.semanticInfoGenerationMode = builder.semanticInfoGenerationMode.getValue();
    }

    @Override
    protected String contentType() {
        return SemanticFieldMapper.CONTENT_TYPE;
    }

    @Override
    public ParametrizedFieldMapper.Builder getMergeBuilder() {
        return new Builder(simpleName(), this.indexCreatedVersion, this.indexAnalyzers).init(this);
    }

    /**
     * Builder for SemanticTextFieldMapper.
     * The builder gathers all necessary field settings before constructing the actual SemanticTextFieldMapper.
     * The builder creates the corresponding MappedFieldType, which controls how queries interact with the field.
     * After setting up all parameters, it builds the actual SemanticTextFieldMapper using the build() method.
     */
    public static class Builder extends TextFieldMapper.Builder {

        protected final Parameter<String> modelId = Parameter.stringParam(MODEL_ID, true, m -> ((SemanticTextFieldMapper) m).modelId, null);
        protected final Parameter<String> rawFieldType = Parameter.stringParam(
            RAW_FIELD_TYPE,
            false,
            m -> ((SemanticTextFieldMapper) m).rawFieldType,
            TextFieldMapper.CONTENT_TYPE
        );
        protected final Parameter<String> semanticInfoFieldName = Parameter.stringParam(
            SEMANTIC_INFO_FIELD_NAME,
            false,
            m -> ((SemanticTextFieldMapper) m).semanticInfoFieldName,
            null
        ).acceptsNull();
        protected final Parameter<String> semanticInfoGenerationMode = Parameter.stringParam(
            SEMANTIC_INFO_GENERATION_MODE,
            true,
            m -> ((SemanticTextFieldMapper) m).semanticInfoGenerationMode,
            SemanticInfoGenerationMode.ALWAYS.getName()
        );

        public Builder(String name, IndexAnalyzers indexAnalyzers) {
            super(name, indexAnalyzers);
        }

        public Builder(String name, Version indexCreatedVersion, IndexAnalyzers indexAnalyzers) {
            super(name, indexCreatedVersion, indexAnalyzers);
        }

        @Override
        protected List<Parameter<?>> getParameters() {
            List<Parameter<?>> parameters = new ArrayList<>();
            for (Parameter<?> p : super.getParameters()) {
                parameters.add(p);
            }
            parameters.add(modelId);
            parameters.add(rawFieldType);
            parameters.add(semanticInfoFieldName);
            parameters.add(semanticInfoGenerationMode);
            return Collections.unmodifiableList(parameters);
        }

        @Override
        public SemanticTextFieldMapper build(BuilderContext builderContext) {
            FieldType fieldType = TextParams.buildFieldType(index, store, indexOptions, norms, termVectors);
            SemanticTextFieldMapper.SemanticTextFieldType tft = buildFieldType(fieldType, builderContext);
            return new SemanticTextFieldMapper(
                name,
                fieldType,
                tft,
                buildPrefixMapper(builderContext, fieldType, tft),
                buildPhraseMapper(fieldType, tft),
                multiFieldsBuilder.build(this, builderContext),
                copyTo.build(),
                this
            );

        }

        @Override
        protected SemanticTextFieldType buildFieldType(FieldType fieldType, BuilderContext context) {
            NamedAnalyzer indexAnalyzer = analyzers.getIndexAnalyzer();
            NamedAnalyzer searchAnalyzer = analyzers.getSearchAnalyzer();
            NamedAnalyzer searchQuoteAnalyzer = analyzers.getSearchQuoteAnalyzer();
            if (positionIncrementGap.get() != POSITION_INCREMENT_GAP_USE_ANALYZER) {
                if (fieldType.indexOptions().compareTo(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS) < 0) {
                    throw new IllegalArgumentException(
                        "Cannot set position_increment_gap on field [" + name + "] without positions enabled"
                    );
                }
                indexAnalyzer = new NamedAnalyzer(indexAnalyzer, positionIncrementGap.get());
                searchAnalyzer = new NamedAnalyzer(searchAnalyzer, positionIncrementGap.get());
                searchQuoteAnalyzer = new NamedAnalyzer(searchQuoteAnalyzer, positionIncrementGap.get());
            }
            TextSearchInfo tsi = new TextSearchInfo(fieldType, similarity.getValue(), searchAnalyzer, searchQuoteAnalyzer);
            SemanticTextFieldType ft = new SemanticTextFieldType(
                buildFullName(context),
                index.getValue(),
                store.getValue(),
                tsi,
                meta.getValue()
            );
            ft.setIndexAnalyzer(indexAnalyzer);
            ft.setEagerGlobalOrdinals(eagerGlobalOrdinals.getValue());
            ft.setBoost(boost.getValue());
            if (fieldData.getValue()) {
                ft.setFielddata(true, freqFilter.getValue());
            }
            ft.setModelId(modelId.getValue());
            ft.setSemanticInfoFieldName(semanticInfoFieldName.getValue());
            return ft;
        }

    }

    /**
     * Parsing the field mapping configuration from JSON when creating or updating an index. It is used to
     * dynamically construct the appropriate FieldMapper based on the field type definition in the mapping.
     */
    public static final TypeParser PARSER = new TypeParser((n, c) -> new Builder(n, c.indexVersionCreated(), c.getIndexAnalyzers()));

    /**
     * Field type for semantic_text fields. It defines the characteristics of a field in the index mapping.
     * It is responsible for storing metadata about how a field should be indexed, stored, and queried.
     */
    @Setter
    @Getter
    public static class SemanticTextFieldType extends TextFieldMapper.TextFieldType {
        private String modelId;
        private String semanticInfoFieldName;

        public SemanticTextFieldType(String name, boolean indexed, boolean stored, TextSearchInfo tsi, Map<String, String> meta) {
            super(name, indexed, stored, tsi, meta);
        }

        @Override
        public String typeName() {
            return SemanticFieldMapper.CONTENT_TYPE;
        }
    }

    @Override
    protected void doXContentBody(XContentBuilder builder, boolean includeDefaults, Params params) throws IOException {
        super.doXContentBody(builder, includeDefaults, params);
        SemanticTextFieldMapper.Builder mapperBuilder = (Builder) getMergeBuilder();
        builder.field(MODEL_ID, mapperBuilder.modelId.getValue());
        builder.field(RAW_FIELD_TYPE, mapperBuilder.rawFieldType.getValue());
        if (mapperBuilder.semanticInfoFieldName.getValue() != null) {
            builder.field(SEMANTIC_INFO_FIELD_NAME, mapperBuilder.semanticInfoFieldName.getValue());
        }
        builder.field(SEMANTIC_INFO_GENERATION_MODE, mapperBuilder.semanticInfoGenerationMode.getValue());
    }

}
