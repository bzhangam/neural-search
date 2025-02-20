/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.mapper;

import lombok.Getter;
import lombok.Setter;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.IndexOptions;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.mapper.DocumentParser;
import org.opensearch.index.mapper.FieldMapper;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.Mapper;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.index.mapper.ObjectMapper;
import org.opensearch.index.mapper.ParametrizedFieldMapper;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.index.mapper.SourceValueFetcher;
import org.opensearch.index.mapper.StringFieldType;
import org.opensearch.index.mapper.TextSearchInfo;
import org.opensearch.index.mapper.ValueFetcher;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.neuralsearch.ml.MLCommonsClientAccessor;
import org.opensearch.search.lookup.SearchLookup;
import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opensearch.neuralsearch.common.SemanticFieldConstants.MODEL_ID;

/**
 * Defining how a semantic_text field is stored, indexed, and queried.
 */
public class SemanticTextFieldMapper extends ParametrizedFieldMapper {

    public static final String CONTENT_TYPE = "semantic_text";
    protected String modelId;
    protected ObjectMapper objectMapper;

    /**
     * Default parameters for semantic_text fields.
     */
    public static class Defaults {
        public static final FieldType INDEXED_FIELD_TYPE = new FieldType();
        public static final FieldType STORE_ONLY_FIELD_TYPE = new FieldType();
        static {
            INDEXED_FIELD_TYPE.setTokenized(true);
            INDEXED_FIELD_TYPE.setStored(false);
            INDEXED_FIELD_TYPE.setStoreTermVectors(false);
            INDEXED_FIELD_TYPE.setOmitNorms(false);
            INDEXED_FIELD_TYPE.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS);
            INDEXED_FIELD_TYPE.freeze();

            STORE_ONLY_FIELD_TYPE.setStored(true);
        }
    }

    protected SemanticTextFieldMapper(
        String simpleName,
        MappedFieldType mappedFieldType,
        MultiFields multiFields,
        CopyTo copyTo,
        Builder builder,
        ObjectMapper objectMapper
    ) {
        super(simpleName, mappedFieldType, multiFields, copyTo);
        this.modelId = builder.modelId.getValue();
        this.objectMapper = objectMapper;
    }

    @Override
    public ParametrizedFieldMapper.Builder getMergeBuilder() {
        return new Builder(simpleName()).init(this);
    }

    @Override
    protected void parseCreateField(ParseContext parseContext) throws IOException {
        XContentParser parser = parseContext.parser();
        XContentParser.Token token = parser.currentToken();
        if (token != XContentParser.Token.START_OBJECT) {
            throw new IOException("Expected a JSON object for semantic text field");
        }
        DocumentParser.parseObjectOrNested(parseContext, objectMapper);

        // index the original text at the path so that we can query it like a normal text
        Field originalTextField = (Field) parseContext.doc().getField(name() + ".original_text");
        String originalText = originalTextField.stringValue();
        Field originalTextFieldAtPath = new Field(name(), originalText, Defaults.INDEXED_FIELD_TYPE);
        parseContext.doc().add(originalTextFieldAtPath);
    }

    @Override
    protected String contentType() {
        return CONTENT_TYPE;
    }

    @Override
    public List<Mapper> getInternalMappers() {
        return List.of(objectMapper);
    }

    @Override
    public boolean allowToBeModeledAsAnObject() {
        return true;
    }

    /**
     * Builder for SemanticTextFieldMapper.
     * The builder gathers all necessary field settings before constructing the actual SemanticTextFieldMapper.
     * The builder creates the corresponding MappedFieldType, which controls how queries interact with the field.
     * After setting up all parameters, it builds the actual SemanticTextFieldMapper using the build() method.
     */
    public static class Builder extends ParametrizedFieldMapper.Builder {

        protected final Parameter<String> modelId = Parameter.stringParam(MODEL_ID, true, m -> ((SemanticTextFieldMapper) m).modelId, null);
        @Setter
        private ObjectMapper.Builder<?> objectMapperBuilder;

        protected Builder(String name) {
            super(name);
        }

        @Override
        protected List<Parameter<?>> getParameters() {
            return List.of(modelId);
        }

        @Override
        public ParametrizedFieldMapper.Builder init(FieldMapper initializer) {
            super.init(initializer);
            // init auto-generated fields
            return this;
        }

        @Override
        public SemanticTextFieldMapper build(BuilderContext builderContext) {
            // We don't support multi fields and copy to for P0
            final MultiFields multiFieldsBuilder = this.multiFieldsBuilder.build(this, builderContext);
            final CopyTo copyToBuilder = copyTo.build();

            final SemanticTextFieldType semanticTextFieldType = new SemanticTextFieldType(buildFullName(builderContext));
            semanticTextFieldType.setModelId(modelId.getValue());
            // Build field mapper
            ObjectMapper objectMapper = objectMapperBuilder.build(builderContext);

            return new SemanticTextFieldMapper(name, semanticTextFieldType, multiFieldsBuilder, copyToBuilder, this, objectMapper);
        }
    }

    /**
     * Parsing the field mapping configuration from JSON when creating or updating an index. It is used to
     * dynamically construct the appropriate FieldMapper based on the field type definition in the mapping.
     */
    public static class TypeParser implements Mapper.TypeParser {
        private final MLCommonsClientAccessor clientAccessor;

        public TypeParser(final MLCommonsClientAccessor clientAccessor) {
            this.clientAccessor = clientAccessor;
        }

        @Override
        public Mapper.Builder<?> parse(String name, Map<String, Object> node, ParserContext parserContext) throws MapperParsingException {
            Builder builder = new Builder(name);
            builder.parse(name, parserContext, node);

            ObjectMapper.TypeParser objectTypeParser = new ObjectMapper.TypeParser();

            // Do some work to pull the model config.

            // We will treat semantic_text as an object field so create the config of subfields for it
            Map<String, Object> config = new HashMap<>();
            Map<String, Object> propertiesConfig = new HashMap<>();

            // for original text treat it as a simple text field
            Map<String, Object> originalTextConfig = new HashMap<>();
            originalTextConfig.put("type", "text");

            // for chunks treat it as a nested object with embeddings
            Map<String, Object> chunksConfig = new HashMap<>();
            chunksConfig.put("type", "nested");

            Map<String, Object> chunksProperties = new HashMap<>();

            Map<String, Object> chunkedTextConfig = new HashMap<>();
            chunkedTextConfig.put("type", "text");

            Map<String, Object> knnConfig = new HashMap<>();
            knnConfig.put("type", "knn_vector");
            knnConfig.put("dimension", 768);
            Map<String, Object> knnMethodConfig = new HashMap<>();
            knnMethodConfig.put("engine", "faiss");
            knnMethodConfig.put("space_type", "l2");
            knnMethodConfig.put("name", "hnsw");
            knnConfig.put("method", knnMethodConfig);

            chunksProperties.put("text", chunkedTextConfig);
            chunksProperties.put("embedding", knnConfig);

            chunksConfig.put("properties", chunksProperties);

            // for model info normal object
            Map<String, Object> modelInfoConfig = new HashMap<>();

            Map<String, Object> modelInfoProperties = new HashMap<>();

            Map<String, Object> modelIdConfig = new HashMap<>();
            modelIdConfig.put("type", "text");
            modelIdConfig.put("index", "false");
            modelIdConfig.put("store", "true");

            Map<String, Object> modelTypeConfig = new HashMap<>();
            modelTypeConfig.put("type", "text");
            modelTypeConfig.put("index", "false");
            modelTypeConfig.put("store", "true");

            modelInfoProperties.put("id", modelIdConfig);
            modelInfoProperties.put("type", modelTypeConfig);

            modelInfoConfig.put("properties", modelInfoProperties);

            // assemble
            propertiesConfig.put("original_text", originalTextConfig);
            propertiesConfig.put("chunks", chunksConfig);
            propertiesConfig.put("model_info", modelInfoConfig);

            config.put("properties", propertiesConfig);

            ObjectMapper.Builder objectBuilder = (ObjectMapper.Builder) objectTypeParser.parse(name, config, parserContext);
            builder.setObjectMapperBuilder(objectBuilder);

            return builder;
        }
    }

    /**
     * Field type for semantic_text fields. It defines the characteristics of a field in the index mapping.
     * It is responsible for storing metadata about how a field should be indexed, stored, and queried.
     */
    public static class SemanticTextFieldType extends StringFieldType {
        @Setter
        @Getter
        private String modelId;

        public SemanticTextFieldType(
            String name,
            boolean isSearchable,
            boolean isStored,
            boolean hasDocValues,
            TextSearchInfo textSearchInfo,
            Map<String, String> meta
        ) {
            super(name, isSearchable, isStored, hasDocValues, textSearchInfo, meta);
        }

        public SemanticTextFieldType(String name) {
            this(
                name,
                true,
                false,
                false,
                new TextSearchInfo(Defaults.INDEXED_FIELD_TYPE, null, Lucene.STANDARD_ANALYZER, Lucene.STANDARD_ANALYZER),
                Collections.emptyMap()
            );
        }

        @Override
        public ValueFetcher valueFetcher(QueryShardContext queryShardContext, SearchLookup searchLookup, String format) {
            // In source the original text is not stored at tha path. Need to specify the full path for it.
            return SourceValueFetcher.toString(name(), queryShardContext, format);
        }

        @Override
        public String typeName() {
            return SemanticTextFieldMapper.CONTENT_TYPE;
        }
    }
}
