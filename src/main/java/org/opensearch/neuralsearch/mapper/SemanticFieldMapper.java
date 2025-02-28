/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.mapper;

import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.Mapper;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.index.mapper.ParametrizedFieldMapper;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.neuralsearch.plugin.NeuralSearch;

import java.io.IOException;
import java.util.Map;

public class SemanticFieldMapper extends ParametrizedFieldMapper {
    public static final String CONTENT_TYPE = "semantic";

    /**
     * Creates a new ParametrizedFieldMapper
     *
     * @param simpleName
     * @param mappedFieldType
     * @param multiFields
     * @param copyTo
     */
    protected SemanticFieldMapper(String simpleName, MappedFieldType mappedFieldType, MultiFields multiFields, CopyTo copyTo) {
        super(simpleName, mappedFieldType, multiFields, copyTo);
        throw new UnsupportedOperationException("Should never be called");
    }

    @Override
    public Builder getMergeBuilder() {
        throw new UnsupportedOperationException("Should never be called");
    }

    @Override
    protected void parseCreateField(ParseContext context) throws IOException {
        throw new UnsupportedOperationException("Should never be called");
    }

    @Override
    protected String contentType() {
        return CONTENT_TYPE;
    }

    public static class TypeParser implements Mapper.TypeParser {
        @Override
        public Mapper.Builder<?> parse(String name, Map<String, Object> node, ParserContext parserContext) throws MapperParsingException {
            String rawFieldType = (String) node.getOrDefault("raw_field_type", "text");
            Mapper.TypeParser typeParser = NeuralSearch.SEMANTIC_MAPPERS.get(CONTENT_TYPE + "_" + rawFieldType);
            if (typeParser == null) {
                throw new IllegalArgumentException("rawFieldType: [" + rawFieldType + "] is not supported");
            }
            return typeParser.parse(name, node, parserContext);
        }
    }
}
