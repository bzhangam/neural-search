/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.processor.semantic;

import com.google.common.annotations.VisibleForTesting;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.core.action.ActionListener;
import org.opensearch.env.Environment;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.analysis.AnalysisRegistry;
import org.opensearch.index.mapper.IndexFieldMapper;
import org.opensearch.ingest.AbstractBatchingProcessor;
import org.opensearch.ingest.IngestDocument;
import org.opensearch.ingest.IngestDocumentWrapper;
import org.opensearch.ml.common.MLModel;
import org.opensearch.neuralsearch.common.SemanticFieldConstants;
import org.opensearch.neuralsearch.ml.MLCommonsClientAccessor;
import org.opensearch.neuralsearch.processor.chunker.Chunker;
import org.opensearch.neuralsearch.processor.chunker.ChunkerFactory;
import org.opensearch.neuralsearch.processor.chunker.FixedTokenLengthChunker;
import org.opensearch.neuralsearch.processor.dto.SemanticFieldInfo;
import org.opensearch.neuralsearch.util.ProcessorDocumentUtils;
import org.opensearch.neuralsearch.util.SemanticMappingUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

import static org.opensearch.neuralsearch.processor.chunker.Chunker.CHUNK_STRING_COUNT_FIELD;
import static org.opensearch.neuralsearch.processor.chunker.Chunker.DEFAULT_MAX_CHUNK_LIMIT;
import static org.opensearch.neuralsearch.processor.chunker.Chunker.MAX_CHUNK_LIMIT_FIELD;

/**
 * Processor for semantic field
 */
@Log4j2
public class SemanticFieldProcessor extends AbstractBatchingProcessor {

    public static final String PROCESSOR_TYPE = "internal_semantic_field";

    Map<String, Map<String, Object>> pathToFieldConfig;
    Map<String, MLModel> modelIdToModelMap;

    protected final MLCommonsClientAccessor mlCommonsClientAccessor;
    private final AnalysisRegistry analysisRegistry;
    private final Environment environment;
    private final ClusterService clusterService;

    private Chunker chunker;

    public SemanticFieldProcessor(
        String tag,
        String description,
        int batchSize,
        Map<String, Map<String, Object>> pathToFieldConfig,
        AnalysisRegistry analysisRegistry,
        MLCommonsClientAccessor mlClientAccessor,
        Environment environment,
        ClusterService clusterService
    ) {
        super(tag, description, batchSize);
        this.pathToFieldConfig = pathToFieldConfig;
        this.mlCommonsClientAccessor = mlClientAccessor;
        this.environment = environment;
        this.clusterService = clusterService;
        this.analysisRegistry = analysisRegistry;
        this.chunker = createChunker();
    }

    private Chunker createChunker() {
        // TODO: Make it configurable later
        Map<String, Object> chunkerParameters = new HashMap<>();
        chunkerParameters.put(FixedTokenLengthChunker.TOKEN_LIMIT_FIELD, 10);
        chunkerParameters.put(FixedTokenLengthChunker.OVERLAP_RATE_FIELD, 0.1);
        chunkerParameters.put(FixedTokenLengthChunker.ANALYSIS_REGISTRY_FIELD, analysisRegistry);
        return ChunkerFactory.create(FixedTokenLengthChunker.ALGORITHM_NAME, chunkerParameters);
    }

    @Override
    public IngestDocument execute(IngestDocument ingestDocument) throws Exception {
        return ingestDocument;
    }

    /**
     * This method will be invoked by PipelineService to make async inference and then delegate the handler to
     * process the inference response or failure.
     * @param ingestDocument {@link IngestDocument} which is the document passed to processor.
     * @param handler {@link BiConsumer} which is the handler which can be used after the inference task is done.
     */
    @Override
    public void execute(IngestDocument ingestDocument, BiConsumer<IngestDocument, Exception> handler) {
        try {
            preprocessIngestDocument(ingestDocument);
            // Collect all the semantic field info based on the path
            List<SemanticFieldInfo> semanticFieldInfoList = getSemanticFieldInfo(ingestDocument);

            if (semanticFieldInfoList.isEmpty()) {
                handler.accept(ingestDocument, null);
            } else {
                if (modelIdToModelMap == null) {
                    fetchModelInfoThenProcess(ingestDocument, semanticFieldInfoList, handler);
                } else {
                    process(ingestDocument, semanticFieldInfoList, handler);
                }
            }
        } catch (Exception e) {
            handler.accept(null, e);
        }
    }

    private void process(
        IngestDocument ingestDocument,
        List<SemanticFieldInfo> semanticFieldInfoList,
        BiConsumer<IngestDocument, Exception> handler
    ) {
        setModelInfo(ingestDocument, semanticFieldInfoList);

        chunk(ingestDocument, semanticFieldInfoList);

        inference(ingestDocument, semanticFieldInfoList, handler);
    }

    private void fetchModelInfoThenProcess(
        IngestDocument ingestDocument,
        List<SemanticFieldInfo> semanticFieldInfoList,
        BiConsumer<IngestDocument, Exception> handler
    ) {
        Set<String> modelIds = SemanticMappingUtils.getUniqueModelIds(pathToFieldConfig);
        modelIdToModelMap = new HashMap<>(modelIds.size());
        final int[] counter = { modelIds.size() };
        for (String modelId : modelIds) {
            mlCommonsClientAccessor.getModel(modelId, ActionListener.wrap(mlModel -> {
                modelIdToModelMap.put(modelId, mlModel);
                if (--counter[0] == 0) {
                    process(ingestDocument, semanticFieldInfoList, handler);
                }
            }, e -> handler.accept(null, e)));
        }
    }

    private void setModelInfo(IngestDocument ingestDocument, List<SemanticFieldInfo> semanticFieldInfoList) {
        Map<String, Map<String, Object>> modelIdToInfoMap = new HashMap<>();
        for (Map.Entry<String, MLModel> entry : modelIdToModelMap.entrySet()) {
            Map<String, Object> modelInfo = new HashMap<>();
            String modelId = entry.getKey();

            MLModel mlModel = entry.getValue();
            modelInfo.put(SemanticFieldConstants.SemanticInfo.ModelInfo.ID, modelId);
            modelInfo.put(SemanticFieldConstants.SemanticInfo.ModelInfo.TYPE, mlModel.getAlgorithm());
            modelInfo.put(SemanticFieldConstants.SemanticInfo.ModelInfo.NAME, mlModel.getName());

            modelIdToInfoMap.put(modelId, modelInfo);
        }

        for (SemanticFieldInfo semanticFieldInfo : semanticFieldInfoList) {
            String modelInfoFullPath = semanticFieldInfo.getSemanticInfoFullPath() + ".model";
            ingestDocument.setFieldValue(modelInfoFullPath, modelIdToInfoMap.get(semanticFieldInfo.getModelId()));
        }
    }

    private void inference(
        IngestDocument ingestDocument,
        List<SemanticFieldInfo> semanticFieldInfoList,
        BiConsumer<IngestDocument, Exception> handler
    ) {
        Map<String, Set<String>> modelIdToValuesMap = new HashMap<>();
        for (SemanticFieldInfo semanticFieldInfo : semanticFieldInfoList) {
            String modelId = semanticFieldInfo.getModelId();
            if (!modelIdToValuesMap.containsKey(modelId)) {
                modelIdToValuesMap.put(modelId, new HashSet<>());
            }
            modelIdToValuesMap.get(modelId).addAll(semanticFieldInfo.getChunks());
        }

        final int[] counter = { modelIdToValuesMap.size() };
        Map<Pair<String, String>, List<Float>> modelIdValueToEmbeddingMap = new HashMap<>();
        for (Map.Entry<String, Set<String>> entry : modelIdToValuesMap.entrySet()) {
            String modelId = entry.getKey();
            List<String> values = new ArrayList<>(entry.getValue());
            mlCommonsClientAccessor.inferenceSentences(entry.getKey(), values, ActionListener.wrap(vectors -> {
                for (int i = 0; i < values.size(); i++) {
                    modelIdValueToEmbeddingMap.put(Pair.of(modelId, values.get(i)), vectors.get(i));
                }
                if (--counter[0] == 0) {
                    setInference(ingestDocument, semanticFieldInfoList, modelIdValueToEmbeddingMap);
                    handler.accept(ingestDocument, null);
                }
            }, e -> { handler.accept(null, e); }));
        }
    }

    private void setInference(
        IngestDocument ingestDocument,
        List<SemanticFieldInfo> semanticFieldInfoList,
        Map<Pair<String, String>, List<Float>> modelIdValueToEmbeddingMap
    ) {
        for (SemanticFieldInfo semanticFieldInfo : semanticFieldInfoList) {
            String modelId = semanticFieldInfo.getModelId();
            List<String> chunks = semanticFieldInfo.getChunks();
            for (int i = 0; i < chunks.size(); i++) {
                String chunk = chunks.get(i);
                List<Float> embedding = modelIdValueToEmbeddingMap.get(Pair.of(modelId, chunk));
                String embeddingFullPath = semanticFieldInfo.getSemanticInfoFullPath() + ".chunks." + i + ".embedding";
                ingestDocument.setFieldValue(embeddingFullPath, embedding);
            }
        }
    }

    private List<SemanticFieldInfo> getSemanticFieldInfo(IngestDocument ingestDocument) {
        List<SemanticFieldInfo> semanticFieldInfos = new ArrayList<>();
        Object doc = ingestDocument.getSourceAndMetadata();
        String rootPath = StringUtils.EMPTY;
        pathToFieldConfig.forEach((path, config) -> {
            Object currentNode = doc;
            collectSemanticFieldInfo(currentNode, path.split("\\."), config, 0, rootPath, semanticFieldInfos);
        });
        return semanticFieldInfos;
    }

    private void chunk(IngestDocument ingestDocument, List<SemanticFieldInfo> semanticFieldInfo) {
        Map<String, Object> sourceAndMetadataMap = ingestDocument.getSourceAndMetadata();
        Map<String, Object> runtimeParameters = new HashMap<>();
        int maxTokenCount = getMaxTokenCount(sourceAndMetadataMap);
        int chunkStringCount = semanticFieldInfo.size();
        runtimeParameters.put(FixedTokenLengthChunker.MAX_TOKEN_COUNT_FIELD, maxTokenCount);
        runtimeParameters.put(MAX_CHUNK_LIMIT_FIELD, DEFAULT_MAX_CHUNK_LIMIT);
        runtimeParameters.put(CHUNK_STRING_COUNT_FIELD, chunkStringCount);
        for (SemanticFieldInfo fieldInfo : semanticFieldInfo) {
            List<String> chunkedText = chunker.chunk(fieldInfo.getValue(), runtimeParameters);
            fieldInfo.setChunks(chunkedText);
            String chunksFullPath = fieldInfo.getFullPathForChunks();
            List<Map<String, Object>> chunks = new ArrayList<>();

            for (String s : chunkedText) {
                Map<String, Object> chunk = new HashMap<>();
                chunk.put("text", s);
                chunks.add(chunk);
            }

            ingestDocument.setFieldValue(chunksFullPath, chunks);
        }
    }

    private int getMaxTokenCount(final Map<String, Object> sourceAndMetadataMap) {
        int defaultMaxTokenCount = IndexSettings.MAX_TOKEN_COUNT_SETTING.get(environment.settings());
        String indexName = sourceAndMetadataMap.get(IndexFieldMapper.NAME).toString();
        IndexMetadata indexMetadata = clusterService.state().metadata().index(indexName);
        if (Objects.isNull(indexMetadata)) {
            return defaultMaxTokenCount;
        }
        // if the index is specified in the metadata, read maxTokenCount from the index setting
        return IndexSettings.MAX_TOKEN_COUNT_SETTING.get(indexMetadata.getSettings());
    }

    /**
     * Recursively collects semantic field values from the document.
     */
    private void collectSemanticFieldInfo(
        Object node,
        String[] pathParts,
        Map<String, Object> fieldConfig,
        int level,
        String currentPath,
        List<SemanticFieldInfo> semanticFieldInfoList
    ) {
        if (level > pathParts.length || node == null) {
            return;
        }

        String key = level < pathParts.length ? pathParts[level] : null;

        if (node instanceof Map<?, ?> mapNode) {
            Object nextNode = mapNode.get(key);
            String newPath = currentPath.isEmpty() ? key : currentPath + "." + key;
            collectSemanticFieldInfo(nextNode, pathParts, fieldConfig, level + 1, newPath, semanticFieldInfoList);
        } else if (node instanceof List<?> listNode) {
            for (int i = 0; i < listNode.size(); i++) {
                Object listItem = listNode.get(i);
                String indexedPath = currentPath + "." + i;
                collectSemanticFieldInfo(listItem, pathParts, fieldConfig, level, indexedPath, semanticFieldInfoList);
            }
        } else if (level == pathParts.length) {
            String modelId = (String) fieldConfig.get(SemanticFieldConstants.MODEL_ID);
            if (modelId == null) {
                throw new IllegalArgumentException(
                    "It does not make sense we try to process the semantic field without a model id. The field is at: "
                        + String.join(".", pathParts)
                );
            }

            String semanticInfoFullPath = currentPath + SemanticFieldConstants.DEFAULT_SEMANTIC_INFO_FIELD_NAME_SUFFIX;
            String userDefinedFieldName = (String) fieldConfig.get(SemanticFieldConstants.SEMANTIC_INFO_FIELD_NAME);
            if (userDefinedFieldName != null) {
                String[] paths = currentPath.split("\\.");
                paths[paths.length - 1] = userDefinedFieldName;
                semanticInfoFullPath = String.join(".", paths);
            }

            SemanticFieldInfo semanticFieldInfo = new SemanticFieldInfo();
            semanticFieldInfo.setValue(node.toString());
            semanticFieldInfo.setModelId(modelId);
            semanticFieldInfo.setFullPath(currentPath);
            semanticFieldInfo.setSemanticInfoFullPath(semanticInfoFullPath);

            semanticFieldInfoList.add(semanticFieldInfo);
        }
    }

    @VisibleForTesting
    void preprocessIngestDocument(IngestDocument ingestDocument) {
        if (ingestDocument == null || ingestDocument.getSourceAndMetadata() == null) return;
        Map<String, Object> sourceAndMetadataMap = ingestDocument.getSourceAndMetadata();
        Map<String, Object> unflattened = ProcessorDocumentUtils.unflattenJson(sourceAndMetadataMap);
        unflattened.forEach(ingestDocument::setFieldValue);
        sourceAndMetadataMap.keySet().removeIf(key -> key.contains("."));
    }

    /**
     * This is the function which does actual inference work for batchExecute interface.
     * @param inferenceList a list of String for inference.
     * @param handler a callback handler to handle inference results which is a list of objects.
     * @param onException an exception callback to handle exception.
     */
    void doBatchExecute(List<String> inferenceList, Consumer<List<?>> handler, Consumer<Exception> onException) {

    }

    @Override
    public void subBatchExecute(List<IngestDocumentWrapper> ingestDocumentWrappers, Consumer<List<IngestDocumentWrapper>> handler) {
        // TODO: Handle the batch case
    }

    @Override
    public String getType() {
        return PROCESSOR_TYPE;
    }

}
