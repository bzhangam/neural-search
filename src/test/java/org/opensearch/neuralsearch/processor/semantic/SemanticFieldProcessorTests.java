/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.processor.semantic;

import org.junit.Before;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.metadata.Metadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.action.ActionListener;
import org.opensearch.env.Environment;
import org.opensearch.index.VersionType;
import org.opensearch.index.analysis.AnalysisRegistry;
import org.opensearch.ingest.IngestDocument;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.model.TextEmbeddingModelConfig;
import org.opensearch.neuralsearch.mapper.SemanticFieldMapper;
import org.opensearch.neuralsearch.ml.MLCommonsClientAccessor;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.neuralsearch.constants.MappingConstants.PATH_SEPARATOR;
import static org.opensearch.neuralsearch.constants.MappingConstants.TYPE;
import static org.opensearch.neuralsearch.constants.SemanticFieldConstants.MODEL_ID;
import static org.opensearch.neuralsearch.processor.TextChunkingProcessorTests.getAnalysisRegistry;

public class SemanticFieldProcessorTests extends OpenSearchTestCase {
    @Mock
    private MLCommonsClientAccessor mlCommonsClientAccessor;

    private AnalysisRegistry analysisRegistry;
    @Mock
    private Environment environment;
    @Mock
    private ClusterService clusterService;
    private final ClassLoader classLoader = this.getClass().getClassLoader();

    private Map<String, Map<String, Object>> pathToFieldConfigMap;

    private SemanticFieldProcessor semanticFieldProcessor;
    private final String DUMMY_MODEL_ID_1 = "dummy_model_id_1";
    private final String DUMMY_MODEL_ID_2 = "dummy_model_id_2";
    private final String FIELD_NAME_PRODUCTS = "products";
    private final String FIELD_NAME_PRODUCT_DESCRIPTION = "product_description";
    private final String FIELD_NAME_GEO_DATA = "geo_data";
    private final String GEO_DATA_1 = "dummy_geo_data_1";

    private MLModel textEmbeddingModel;
    private MLModel sparseEmbeddingModel;

    @Before
    public void setup() {
        MockitoAnnotations.openMocks(this);
        // mock env
        final Settings settings = Settings.builder()
            .put("index.mapping.depth.limit", 20)
            .put("index.analyze.max_token_count", 10000)
            .put("index.number_of_shards", 1)
            .build();
        when(environment.settings()).thenReturn(settings);
        // mock cluster
        final Metadata metadata = mock(Metadata.class);
        final ClusterState clusterState = mock(ClusterState.class);
        final ClusterService clusterService = mock(ClusterService.class);
        when(metadata.index(anyString())).thenReturn(null);
        when(clusterState.metadata()).thenReturn(metadata);
        when(clusterService.state()).thenReturn(clusterState);
        // mock analysisRegistry
        analysisRegistry = getAnalysisRegistry();

        // two semantic fields with different model ids
        pathToFieldConfigMap = Map.of(
            FIELD_NAME_PRODUCTS + PATH_SEPARATOR + FIELD_NAME_PRODUCT_DESCRIPTION,
            Map.of(TYPE, SemanticFieldMapper.CONTENT_TYPE, MODEL_ID, DUMMY_MODEL_ID_1),
            FIELD_NAME_GEO_DATA,
            Map.of(TYPE, SemanticFieldMapper.CONTENT_TYPE, MODEL_ID, DUMMY_MODEL_ID_2)
        );

        // prepare mock model config
        final Integer embeddingDimension = 768;
        final String allConfig = "{\"space_type\":\"l2\"}";
        final TextEmbeddingModelConfig textEmbeddingModelConfig = TextEmbeddingModelConfig.builder()
            .embeddingDimension(embeddingDimension)
            .allConfig(allConfig)
            .modelType("modelType")
            .frameworkType(TextEmbeddingModelConfig.FrameworkType.HUGGINGFACE_TRANSFORMERS)
            .build();
        textEmbeddingModel = MLModel.builder()
            .algorithm(FunctionName.TEXT_EMBEDDING)
            .modelConfig(textEmbeddingModelConfig)
            .name(FunctionName.TEXT_EMBEDDING.name())
            .build();

        sparseEmbeddingModel = MLModel.builder().algorithm(FunctionName.SPARSE_ENCODING).name(FunctionName.SPARSE_ENCODING.name()).build();

        semanticFieldProcessor = new SemanticFieldProcessor(
            "tag",
            "description",
            1,
            pathToFieldConfigMap,
            analysisRegistry,
            mlCommonsClientAccessor,
            environment,
            clusterService
        );
    }

    public void testExecute_whenNoSemanticField_thenDoNothing() {
        // Scenario: No semantic fields in the ingest document
        final IngestDocument ingestDocument = new IngestDocument("index", "1", "routing", 1L, VersionType.INTERNAL, new HashMap<>());

        // Call the method
        semanticFieldProcessor.execute(ingestDocument, (doc, e) -> {
            assertNull("No error should occur", e);
            assertNotNull("Ingest document should be passed unchanged", doc);
        });
    }

    public void testExecute_whenValidDoc_thenIngestDoc() throws URISyntaxException, IOException {

        // prepare ingest doc
        final Map<String, Object> ingestDocSource = readDocSourceFromFile("processor/semantic/ingest_doc1.json");
        final IngestDocument ingestDocument = new IngestDocument("index", "1", "routing", 1L, VersionType.INTERNAL, ingestDocSource);

        // mock get model API
        doAnswer(invocationOnMock -> {
            final String modelId = invocationOnMock.getArgument(0);
            final ActionListener<MLModel> listener = invocationOnMock.getArgument(1);
            if (DUMMY_MODEL_ID_1.equals(modelId)) {
                listener.onResponse(textEmbeddingModel);
            } else if (DUMMY_MODEL_ID_2.equals(modelId)) {
                listener.onResponse(sparseEmbeddingModel);
            } else {
                listener.onFailure(new RuntimeException("Model not found"));
            }
            return null;
        }).when(mlCommonsClientAccessor).getModel(any(), any());

        // Call the method
        semanticFieldProcessor.execute(ingestDocument, (doc, e) -> {
            assertNull("No error should occur", e);
            assertNotNull("Ingest document should be passed unchanged", doc);
        });
    }

    private Map<String, Object> readDocSourceFromFile(String filePath) throws URISyntaxException, IOException {
        final String docStr = Files.readString(Path.of(classLoader.getResource(filePath).toURI()));
        return XContentHelper.convertToMap(XContentType.JSON.xContent(), docStr, false);
    }

}
