/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.processor.semantic;

import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.Nullable;
import org.opensearch.core.action.ActionListener;
import org.opensearch.env.Environment;
import org.opensearch.index.analysis.AnalysisRegistry;
import org.opensearch.ingest.AbstractBatchingSystemProcessor;
import org.opensearch.ingest.IngestDocument;
import org.opensearch.ingest.IngestDocumentWrapper;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.model.MLModelConfig;
import org.opensearch.neuralsearch.ml.MLCommonsClientAccessor;
import org.opensearch.neuralsearch.processor.TextInferenceRequest;
import org.opensearch.neuralsearch.processor.chunker.Chunker;
import org.opensearch.neuralsearch.processor.chunker.ChunkerFactory;
import org.opensearch.neuralsearch.processor.chunker.FixedTokenLengthChunker;
import org.opensearch.neuralsearch.processor.dto.SemanticFieldInfo;
import org.opensearch.neuralsearch.util.TokenWeightUtil;
import org.opensearch.neuralsearch.util.prune.PruneType;
import org.opensearch.neuralsearch.util.prune.PruneUtils;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

import static org.opensearch.neuralsearch.constants.MappingConstants.PATH_SEPARATOR;
import static org.opensearch.neuralsearch.constants.SemanticFieldConstants.DEFAULT_SEMANTIC_INFO_FIELD_NAME_SUFFIX;
import static org.opensearch.neuralsearch.constants.SemanticFieldConstants.MODEL_ID;
import static org.opensearch.neuralsearch.constants.SemanticFieldConstants.SEMANTIC_INFO_FIELD_NAME;
import static org.opensearch.neuralsearch.constants.SemanticInfoFieldConstants.CHUNKS_TEXT_FIELD_NAME;
import static org.opensearch.neuralsearch.constants.SemanticInfoFieldConstants.MODEL_ID_FIELD_NAME;
import static org.opensearch.neuralsearch.constants.SemanticInfoFieldConstants.MODEL_NAME_FIELD_NAME;
import static org.opensearch.neuralsearch.constants.SemanticInfoFieldConstants.MODEL_TYPE_FIELD_NAME;
import static org.opensearch.neuralsearch.mappingtransformer.SemanticMappingTransformer.SUPPORTED_MODEL_ALGORITHM;
import static org.opensearch.neuralsearch.mappingtransformer.SemanticMappingTransformer.SUPPORTED_REMOTE_MODEL_TYPES;
import static org.opensearch.neuralsearch.processor.chunker.Chunker.CHUNK_STRING_COUNT_FIELD;
import static org.opensearch.neuralsearch.processor.chunker.Chunker.DEFAULT_MAX_CHUNK_LIMIT;
import static org.opensearch.neuralsearch.processor.chunker.Chunker.MAX_CHUNK_LIMIT_FIELD;
import static org.opensearch.neuralsearch.processor.util.ProcessorUtils.getMaxTokenCount;
import static org.opensearch.neuralsearch.util.ProcessorDocumentUtils.unflattenIngestDoc;
import static org.opensearch.neuralsearch.util.SemanticMappingUtils.getUniqueModelIds;

/**
 * Processor to ingest the semantic fields.
 *
 * This processor is for internal usage and will be systematically invoked if we detect there are semantic fields
 * defined. Users should not be able to define this processor in an ingest pipeline.
 */
@Log4j2
public class SemanticFieldProcessor extends AbstractBatchingSystemProcessor {

    public static final String PROCESSOR_TYPE = "index_based_ingest_processor_semantic_field";

    private final Map<String, Map<String, Object>> pathToFieldConfig;
    private final Map<String, MLModel> modelIdToModelMap = new HashMap<>();
    private final Map<String, String> modelIdToModelTypeMap = new HashMap<>();

    protected final MLCommonsClientAccessor mlCommonsClientAccessor;
    private final AnalysisRegistry analysisRegistry;
    private final Environment environment;
    private final ClusterService clusterService;

    private final Chunker chunker;

    public SemanticFieldProcessor(
        @Nullable final String tag,
        @Nullable final String description,
        final int batchSize,
        @NonNull final Map<String, Map<String, Object>> pathToFieldConfig,
        @NonNull final AnalysisRegistry analysisRegistry,
        @NonNull final MLCommonsClientAccessor mlClientAccessor,
        @NonNull final Environment environment,
        @NonNull final ClusterService clusterService
    ) {
        super(tag, description, batchSize);
        this.pathToFieldConfig = pathToFieldConfig;
        this.mlCommonsClientAccessor = mlClientAccessor;
        this.environment = environment;
        this.clusterService = clusterService;
        this.analysisRegistry = analysisRegistry;
        this.chunker = createChunker();
    }

    /**
     * Create a default text chunker.
     * TODO: Make it configurable
     * @return A default fixed token length chunker
     */
    private Chunker createChunker() {
        Map<String, Object> chunkerParameters = new HashMap<>();
        chunkerParameters.put(FixedTokenLengthChunker.TOKEN_LIMIT_FIELD, 50);
        chunkerParameters.put(FixedTokenLengthChunker.OVERLAP_RATE_FIELD, 0.2);
        chunkerParameters.put(FixedTokenLengthChunker.ANALYSIS_REGISTRY_FIELD, analysisRegistry);
        return ChunkerFactory.create(FixedTokenLengthChunker.ALGORITHM_NAME, chunkerParameters);
    }

    /**
     * Since we need to do async work in this processor we will not invoke this function to ingest the doc.
     * So in this function we simply return the doc directly so that it will not be dropped.
     * @param ingestDocument {@link IngestDocument} which is the document passed to processor.
     * @return {@link IngestDocument} document unchanged
     */
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
            unflattenIngestDoc(ingestDocument);
            // Collect all the semantic field info based on the path of semantic fields found in the index mapping
            final List<SemanticFieldInfo> semanticFieldInfoList = getSemanticFieldInfo(ingestDocument);

            if (semanticFieldInfoList.isEmpty()) {
                handler.accept(ingestDocument, null);
            } else {
                fetchModelInfoThenProcess(ingestDocument, semanticFieldInfoList, handler);
            }
        } catch (Exception e) {
            handler.accept(null, e);
        }
    }

    private void process(
        @NonNull final IngestDocument ingestDocument,
        @NonNull final List<SemanticFieldInfo> semanticFieldInfoList,
        @NonNull final BiConsumer<IngestDocument, Exception> handler
    ) {
        setModelInfo(ingestDocument, semanticFieldInfoList);

        chunk(ingestDocument, semanticFieldInfoList);

        generateAndSetEmbedding(ingestDocument, semanticFieldInfoList, handler);
    }

    private void fetchModelInfoThenProcess(
        @NonNull final IngestDocument ingestDocument,
        @NonNull final List<SemanticFieldInfo> semanticFieldInfoList,
        @NonNull final BiConsumer<IngestDocument, Exception> handler
    ) {
        final Set<String> modelIdsToGetConfig = getUniqueModelIds(pathToFieldConfig);
        // In P0 we do not handle the case that we need re-pull the model config if it's updated through the
        // ml-common update model API. If the update is not backward compatible we will throw exception in later
        // phase. e.g. If the embedding dimension of a dense model is changed and not match the one defined in the
        // index mapping we will fail indexing doc and query dense embedding.
        //
        // In most of the cases we do not change the index config and model config frequently and cache the
        // model config here can have a better performance to ingest the semantic fields.
        //
        // If the index itself is updated e.g. adding another semantic field we will recreate the semantic field
        // processor with the latest semantic field config. This is controlled in core.
        for (final String existingModelId : modelIdToModelMap.keySet()) {
            modelIdsToGetConfig.remove(existingModelId);
        }
        if (modelIdsToGetConfig.isEmpty()) {
            process(ingestDocument, semanticFieldInfoList, handler);
        } else {
            final AtomicInteger counter = new AtomicInteger(modelIdsToGetConfig.size());
            for (final String modelId : modelIdsToGetConfig) {
                mlCommonsClientAccessor.getModel(modelId, ActionListener.wrap(mlModel -> {
                    modelIdToModelMap.put(modelId, mlModel);
                    if (counter.decrementAndGet() == 0) {
                        process(ingestDocument, semanticFieldInfoList, handler);
                    }
                }, e -> handler.accept(null, e)));
            }
        }
    }

    private void setModelInfo(@NonNull final IngestDocument ingestDocument, @NonNull final List<SemanticFieldInfo> semanticFieldInfoList) {
        final Map<String, Map<String, Object>> modelIdToInfoMap = new HashMap<>();
        for (final Map.Entry<String, MLModel> entry : modelIdToModelMap.entrySet()) {
            final Map<String, Object> modelInfo = new HashMap<>();
            final String modelId = entry.getKey();
            final MLModel mlModel = entry.getValue();

            String modelType;
            if (modelIdToModelTypeMap.containsKey(modelId)) {
                modelType = modelIdToModelTypeMap.get(modelId);
            } else {
                modelType = getModelType(mlModel);
                modelIdToModelTypeMap.put(modelId, modelType);
            }

            modelInfo.put(MODEL_ID_FIELD_NAME, modelId);
            modelInfo.put(MODEL_TYPE_FIELD_NAME, modelType);
            modelInfo.put(MODEL_NAME_FIELD_NAME, mlModel.getName());

            modelIdToInfoMap.put(modelId, modelInfo);
        }

        for (final SemanticFieldInfo semanticFieldInfo : semanticFieldInfoList) {
            ingestDocument.setFieldValue(semanticFieldInfo.getFullPathForModelInfo(), modelIdToInfoMap.get(semanticFieldInfo.getModelId()));
        }
    }

    private String getModelType(@NonNull final MLModel mlModel) {
        final FunctionName functionName = mlModel.getAlgorithm();
        final String modelId = mlModel.getModelId();
        String modelType;

        switch (functionName) {
            case FunctionName.TEXT_EMBEDDING:
            case FunctionName.SPARSE_ENCODING:
            case FunctionName.SPARSE_TOKENIZE:
                modelType = functionName.name();
                break;
            case FunctionName.REMOTE:
                final MLModelConfig remoteModelConfig = mlModel.getModelConfig();
                if (remoteModelConfig == null) {
                    throw new IllegalArgumentException("model config of the remote model " + modelId + " is null");
                }
                final String remoteModelType = remoteModelConfig.getModelType();
                final FunctionName modelTypeFunctionName;
                final String errMsgUnsupportedRemoteModelType = "remote model type is not supported for model id "
                    + modelId
                    + ". It should be one of ["
                    + String.join(", ", SUPPORTED_REMOTE_MODEL_TYPES)
                    + "].";
                try {
                    modelTypeFunctionName = FunctionName.from(remoteModelType);
                } catch (IllegalArgumentException e) {
                    throw new IllegalArgumentException(errMsgUnsupportedRemoteModelType);
                }
                modelType = switch (modelTypeFunctionName) {
                    case TEXT_EMBEDDING, SPARSE_ENCODING, SPARSE_TOKENIZE -> FunctionName.REMOTE.name()
                        + "_"
                        + modelTypeFunctionName.name();
                    default -> throw new IllegalArgumentException(errMsgUnsupportedRemoteModelType);
                };
                break;
            default:
                final String errMsgUnsupportedModelType = "model type is not supported for model id "
                    + modelId
                    + ". It should be one of ["
                    + String.join(", ", SUPPORTED_MODEL_ALGORITHM)
                    + "].";
                throw new IllegalArgumentException(errMsgUnsupportedModelType);
        }
        return modelType;
    }

    @SuppressWarnings("unchecked")
    private void generateAndSetEmbedding(
        @NonNull final IngestDocument ingestDocument,
        @NonNull final List<SemanticFieldInfo> semanticFieldInfoList,
        @NonNull final BiConsumer<IngestDocument, Exception> handler
    ) {
        final Map<String, Set<String>> modelIdToRawDataMap = groupRawDataByModelId(semanticFieldInfoList);

        doGenerateAndSetEmbedding(modelIdToRawDataMap, modelIdValueToEmbeddingMap -> {
            setInference(ingestDocument, semanticFieldInfoList, modelIdValueToEmbeddingMap);
            handler.accept(ingestDocument, null);
        }, e -> handler.accept(null, e));
    }

    private Map<String, Set<String>> groupRawDataByModelId(@NonNull final Collection<List<SemanticFieldInfo>> semanticFieldInfoLists) {
        final Map<String, Set<String>> modelIdToRawDataMap = new HashMap<>();
        for (final List<SemanticFieldInfo> semanticFieldInfoList : semanticFieldInfoLists) {
            for (final SemanticFieldInfo semanticFieldInfo : semanticFieldInfoList) {
                modelIdToRawDataMap.computeIfAbsent(semanticFieldInfo.getModelId(), k -> new HashSet<>())
                    .addAll(semanticFieldInfo.getChunks());
            }
        }
        return modelIdToRawDataMap;
    }

    private Map<String, Set<String>> groupRawDataByModelId(@NonNull final List<SemanticFieldInfo> semanticFieldInfoList) {
        return groupRawDataByModelId(Collections.singleton(semanticFieldInfoList));
    }

    private boolean isDenseModel(@NonNull final String modelType) {
        return FunctionName.TEXT_EMBEDDING.name().equals(modelType)
            || (FunctionName.REMOTE.name() + "_" + FunctionName.TEXT_EMBEDDING.name()).equals(modelType);
    }

    @SuppressWarnings("unchecked")
    private void setInference(
        @NonNull final IngestDocument ingestDocument,
        @NonNull final List<SemanticFieldInfo> semanticFieldInfoList,
        @NonNull final Map<Pair<String, String>, Object> modelIdValueToEmbeddingMap
    ) {
        for (final SemanticFieldInfo semanticFieldInfo : semanticFieldInfoList) {
            final String modelId = semanticFieldInfo.getModelId();
            final boolean isDenseModel = isDenseModel(modelIdToModelTypeMap.get(modelId));
            final List<String> chunks = semanticFieldInfo.getChunks();
            for (int i = 0; i < chunks.size(); i++) {
                final String chunk = chunks.get(i);
                Object embedding = modelIdValueToEmbeddingMap.get(Pair.of(modelId, chunk));
                // TODO: In future we should allow user to configure how we should prune the sparse embedding
                // for each semantic field. Then we can pull the config from the semantic config and use it here.
                if (!isDenseModel) {
                    embedding = PruneUtils.pruneSparseVector(PruneType.MAX_RATIO, 0.1f, (Map<String, Float>) embedding);
                }
                final String embeddingFullPath = semanticFieldInfo.getFullPathForEmbedding(i);
                ingestDocument.setFieldValue(embeddingFullPath, embedding);
            }
        }
    }

    private List<SemanticFieldInfo> getSemanticFieldInfo(IngestDocument ingestDocument) {
        final List<SemanticFieldInfo> semanticFieldInfos = new ArrayList<>();
        final Object doc = ingestDocument.getSourceAndMetadata();
        final String rootPath = StringUtils.EMPTY;
        pathToFieldConfig.forEach(
            (path, config) -> { collectSemanticFieldInfo(doc, path.split("\\."), config, 0, rootPath, semanticFieldInfos); }
        );
        return semanticFieldInfos;
    }

    private void chunk(@NonNull final IngestDocument ingestDocument, @NonNull final List<SemanticFieldInfo> semanticFieldInfo) {
        final Map<String, Object> sourceAndMetadataMap = ingestDocument.getSourceAndMetadata();
        final Map<String, Object> runtimeParameters = new HashMap<>();
        int maxTokenCount = getMaxTokenCount(sourceAndMetadataMap, environment.settings(), clusterService);
        int chunkStringCount = semanticFieldInfo.size();
        runtimeParameters.put(FixedTokenLengthChunker.MAX_TOKEN_COUNT_FIELD, maxTokenCount);
        runtimeParameters.put(MAX_CHUNK_LIMIT_FIELD, DEFAULT_MAX_CHUNK_LIMIT);
        runtimeParameters.put(CHUNK_STRING_COUNT_FIELD, chunkStringCount);
        for (final SemanticFieldInfo fieldInfo : semanticFieldInfo) {
            final List<String> chunkedText = chunker.chunk(fieldInfo.getValue(), runtimeParameters);
            fieldInfo.setChunks(chunkedText);
            final String chunksFullPath = fieldInfo.getFullPathForChunks();
            final List<Map<String, Object>> chunks = new ArrayList<>();

            for (final String s : chunkedText) {
                final Map<String, Object> chunk = new HashMap<>();
                chunk.put(CHUNKS_TEXT_FIELD_NAME, s);
                chunks.add(chunk);
            }

            ingestDocument.setFieldValue(chunksFullPath, chunks);
        }
    }

    /**
     * Recursively collects semantic field values from the document.
     */
    private void collectSemanticFieldInfo(
        @Nullable final Object node,
        @NonNull final String[] pathParts,
        @NonNull final Map<String, Object> fieldConfig,
        final int level,
        @NonNull final String currentPath,
        @NonNull final List<SemanticFieldInfo> semanticFieldInfoList
    ) {
        if (level > pathParts.length || node == null) {
            return;
        }

        final String key = level < pathParts.length ? pathParts[level] : null;

        if (level < pathParts.length && node instanceof Map<?, ?> mapNode) {
            final Object nextNode = mapNode.get(key);
            final String newPath = currentPath.isEmpty() ? key : currentPath + PATH_SEPARATOR + key;
            collectSemanticFieldInfo(nextNode, pathParts, fieldConfig, level + 1, newPath, semanticFieldInfoList);
        } else if (level < pathParts.length && node instanceof List<?> listNode) {
            for (int i = 0; i < listNode.size(); i++) {
                final Object listItem = listNode.get(i);
                final String indexedPath = currentPath + PATH_SEPARATOR + i;
                collectSemanticFieldInfo(listItem, pathParts, fieldConfig, level, indexedPath, semanticFieldInfoList);
            }
        } else if (level == pathParts.length) {
            if (!(node instanceof String)) {
                throw new IllegalArgumentException(
                    "Expect the semantic field at path: "
                        + String.join(PATH_SEPARATOR, pathParts)
                        + " to be a string but found: "
                        + node.getClass()
                );
            }
            final String modelId = (String) fieldConfig.get(MODEL_ID);
            if (modelId == null) {
                throw new IllegalArgumentException(
                    "It does not make sense we try to process the semantic field without a model id. The field is at: "
                        + String.join(PATH_SEPARATOR, pathParts)
                );
            }

            String semanticInfoFullPath = currentPath + DEFAULT_SEMANTIC_INFO_FIELD_NAME_SUFFIX;
            final String userDefinedFieldName = (String) fieldConfig.get(SEMANTIC_INFO_FIELD_NAME);
            if (userDefinedFieldName != null) {
                final String[] paths = currentPath.split("\\.");
                paths[paths.length - 1] = userDefinedFieldName;
                semanticInfoFullPath = String.join(PATH_SEPARATOR, paths);
            }

            final SemanticFieldInfo semanticFieldInfo = new SemanticFieldInfo();
            semanticFieldInfo.setValue(node.toString());
            semanticFieldInfo.setModelId(modelId);
            semanticFieldInfo.setFullPath(currentPath);
            semanticFieldInfo.setSemanticInfoFullPath(semanticInfoFullPath);

            semanticFieldInfoList.add(semanticFieldInfo);
        }
    }

    @Override
    public void subBatchExecute(List<IngestDocumentWrapper> ingestDocumentWrappers, Consumer<List<IngestDocumentWrapper>> handler) {
        if (ingestDocumentWrappers == null || ingestDocumentWrappers.isEmpty()) {
            handler.accept(ingestDocumentWrappers);
            return;
        }
        final Map<IngestDocumentWrapper, List<SemanticFieldInfo>> docToSemanticFieldInfoMap = new HashMap<>();
        for (final IngestDocumentWrapper ingestDocumentWrapper : ingestDocumentWrappers) {
            final IngestDocument ingestDocument = ingestDocumentWrapper.getIngestDocument();
            if (ingestDocument == null) {
                continue;
            }
            unflattenIngestDoc(ingestDocument);
            final List<SemanticFieldInfo> semanticFieldInfoList = getSemanticFieldInfo(ingestDocument);
            if (!semanticFieldInfoList.isEmpty()) {
                docToSemanticFieldInfoMap.put(ingestDocumentWrapper, semanticFieldInfoList);
            }
        }

        if (docToSemanticFieldInfoMap.isEmpty()) {
            handler.accept(ingestDocumentWrappers);
        } else {
            fetchModelInfoThenBatchProcess(ingestDocumentWrappers, docToSemanticFieldInfoMap, handler);
        }
    }

    private void fetchModelInfoThenBatchProcess(
        @NonNull final List<IngestDocumentWrapper> ingestDocumentWrappers,
        @NonNull final Map<IngestDocumentWrapper, List<SemanticFieldInfo>> docToSemanticFieldInfoMap,
        @NonNull final Consumer<List<IngestDocumentWrapper>> handler
    ) {
        final Set<String> modelIdsToGetConfig = getUniqueModelIds(pathToFieldConfig);
        for (String existingModelId : modelIdToModelMap.keySet()) {
            modelIdsToGetConfig.remove(existingModelId);
        }
        if (modelIdsToGetConfig.isEmpty()) {
            batchProcess(ingestDocumentWrappers, docToSemanticFieldInfoMap, handler);
        } else {
            final AtomicInteger counter = new AtomicInteger(modelIdsToGetConfig.size());
            final AtomicBoolean hasError = new AtomicBoolean(false);
            for (String modelId : modelIdsToGetConfig) {
                mlCommonsClientAccessor.getModel(modelId, ActionListener.wrap(mlModel -> {
                    modelIdToModelMap.put(modelId, mlModel);
                    if (counter.decrementAndGet() == 0) {
                        if (hasError.get()) {
                            // If we fail to find one model we fail the whole request to keep things simple.
                            handler.accept(ingestDocumentWrappers);
                        } else {
                            batchProcess(ingestDocumentWrappers, docToSemanticFieldInfoMap, handler);
                        }
                    }
                }, e -> {
                    hasError.set(true);
                    addExceptionToImpactedDocs(docToSemanticFieldInfoMap.keySet(), e);
                    if (counter.decrementAndGet() == 0) {
                        handler.accept(ingestDocumentWrappers);
                    }
                }));
            }
        }
    }

    private void batchProcess(
        @NonNull final List<IngestDocumentWrapper> ingestDocumentWrappers,
        @NonNull final Map<IngestDocumentWrapper, List<SemanticFieldInfo>> docToSemanticFieldInfoMap,
        @NonNull final Consumer<List<IngestDocumentWrapper>> handler
    ) {

        for (Map.Entry<IngestDocumentWrapper, List<SemanticFieldInfo>> entry : docToSemanticFieldInfoMap.entrySet()) {
            final IngestDocumentWrapper ingestDocumentWrapper = entry.getKey();
            final IngestDocument ingestDocument = entry.getKey().getIngestDocument();
            final List<SemanticFieldInfo> semanticFieldInfoList = entry.getValue();
            try {
                setModelInfo(ingestDocument, semanticFieldInfoList);

                chunk(ingestDocument, semanticFieldInfoList);
            } catch (Exception e) {
                if (ingestDocumentWrapper.getException() == null) {
                    ingestDocumentWrapper.update(ingestDocument, e);
                }
            }
        }

        batchGenerateAndSetEmbedding(ingestDocumentWrappers, docToSemanticFieldInfoMap, handler);
    }

    @SuppressWarnings("unchecked")
    private void doGenerateAndSetEmbedding(
        @NonNull final Map<String, Set<String>> modelIdToRawDataMap,
        @NonNull final Consumer<Map<Pair<String, String>, Object>> onSuccess,
        @NonNull final Consumer<Exception> onFailure
    ) {
        final AtomicInteger counter = new AtomicInteger(modelIdToRawDataMap.size());
        final AtomicBoolean hasError = new AtomicBoolean(false);
        final List<String> errors = new ArrayList<>();
        final Map<Pair<String, String>, Object> modelIdValueToEmbeddingMap = new HashMap<>();

        for (final Map.Entry<String, Set<String>> entry : modelIdToRawDataMap.entrySet()) {
            final String modelId = entry.getKey();
            final boolean isDenseModel = isDenseModel(modelIdToModelTypeMap.get(modelId));
            final List<String> values = new ArrayList<>(entry.getValue());

            final TextInferenceRequest textInferenceRequest = TextInferenceRequest.builder().inputTexts(values).modelId(modelId).build();

            final ActionListener<?> listener = ActionListener.wrap(embeddings -> {
                List<?> formattedEmbeddings = (List<?>) embeddings;
                if (!isDenseModel) {
                    formattedEmbeddings = TokenWeightUtil.fetchListOfTokenWeightMap((List<Map<String, ?>>) embeddings);
                }
                for (int i = 0; i < values.size(); i++) {
                    modelIdValueToEmbeddingMap.put(Pair.of(modelId, values.get(i)), formattedEmbeddings.get(i));
                }
                if (counter.decrementAndGet() == 0) {
                    if (hasError.get()) {
                        // If we fail to inference any field simply fail the whole request for simplicity
                        onFailure.accept(new RuntimeException(String.join(";", errors)));
                    } else {
                        onSuccess.accept(modelIdValueToEmbeddingMap);
                    }
                }
            }, e -> {
                hasError.set(true);
                errors.add(e.getMessage());
                if (counter.decrementAndGet() == 0) {
                    onFailure.accept(new RuntimeException(String.join(";", errors)));
                }
            });

            if (isDenseModel) {
                mlCommonsClientAccessor.inferenceSentences(textInferenceRequest, (ActionListener<List<List<Number>>>) listener);
            } else {
                mlCommonsClientAccessor.inferenceSentencesWithMapResult(
                    textInferenceRequest,
                    (ActionListener<List<Map<String, ?>>>) listener
                );
            }
        }
    }

    private void batchGenerateAndSetEmbedding(
        @NonNull final List<IngestDocumentWrapper> ingestDocumentWrappers,
        @NonNull final Map<IngestDocumentWrapper, List<SemanticFieldInfo>> docToSemanticFieldInfoMap,
        @NonNull final Consumer<List<IngestDocumentWrapper>> handler
    ) {
        final Map<String, Set<String>> modelIdToRawDataMap = groupRawDataByModelId(docToSemanticFieldInfoMap.values());

        doGenerateAndSetEmbedding(modelIdToRawDataMap, modelIdValueToEmbeddingMap -> {
            batchSetInference(docToSemanticFieldInfoMap, modelIdValueToEmbeddingMap);
            handler.accept(ingestDocumentWrappers);
        }, e -> {
            addExceptionToImpactedDocs(docToSemanticFieldInfoMap.keySet(), e);
            handler.accept(ingestDocumentWrappers);
        });
    }

    private void batchSetInference(
        @NonNull final Map<IngestDocumentWrapper, List<SemanticFieldInfo>> docToSemanticFieldInfoMap,
        @NonNull final Map<Pair<String, String>, Object> modelIdValueToEmbeddingMap
    ) {
        for (Map.Entry<IngestDocumentWrapper, List<SemanticFieldInfo>> entry : docToSemanticFieldInfoMap.entrySet()) {
            final IngestDocument ingestDocument = entry.getKey().getIngestDocument();
            final List<SemanticFieldInfo> semanticFieldInfoList = entry.getValue();
            setInference(ingestDocument, semanticFieldInfoList, modelIdValueToEmbeddingMap);
        }
    }

    private void addExceptionToImpactedDocs(@NonNull final Set<IngestDocumentWrapper> impactedDocs, @NonNull final Exception e) {

        for (final IngestDocumentWrapper ingestDocumentWrapper : impactedDocs) {
            // Do not override the previous exception. We do not filter out the doc with exception at the
            // beginning because previous processor may want to ignore the failure which means even the doc
            // already run into some exception we still want to apply this processor.
            //
            // Ideally we should persist all the exceptions the doc run into so that user can view them
            // together rather than fix one then retry and run into another. This is something we can
            // enhance in the future.
            if (ingestDocumentWrapper.getException() == null) {
                ingestDocumentWrapper.update(ingestDocumentWrapper.getIngestDocument(), e);
            }
        }

    }

    @Override
    public String getType() {
        return PROCESSOR_TYPE;
    }
}
