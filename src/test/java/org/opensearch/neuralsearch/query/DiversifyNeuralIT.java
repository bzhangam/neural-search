/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.query;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.neuralsearch.BaseNeuralSearchIT;

import java.util.List;
import java.util.Map;

/**
 * Verifies the {@code diversify} (MMR) retriever works over a {@code neural} query leg end-to-end, with
 * <b>no neural-search code change</b>: a {@code neural} query matches documents whose embeddings live in a
 * top-level {@code knn_vector} field, and {@code diversify} pulls those same vectors via the k-NN
 * {@code docvalue_fields} ride-along to run MMR. neural-search contributes no MMR logic.
 * <p>
 * CI-only: requires a loaded text-embedding model (downloaded + ML inference), so it runs in the
 * neural-search integTest environment, not on the Faiss-only local dev host.
 * <p>
 * Scope note: this covers a <b>directly-mapped</b> {@code knn_vector} field (the classic text_embedding +
 * ingest-pipeline pattern). The newer {@code semantic} field type stores its vector at a nested
 * {@code <field>_semantic_info.embedding} path; diversify v1 does not support a nested vector path (the
 * shared {@code MMRUtil} field resolution rejects nested) — see the LLD limitation note.
 */
public class DiversifyNeuralIT extends BaseNeuralSearchIT {

    private static final String INDEX = "diversify_neural_index";
    private static final String PIPELINE = "diversify_neural_pipeline";
    private static final String TEXT_FIELD = "text";
    private static final String EMBEDDING_FIELD = "text_knn";

    @SneakyThrows
    public void testDiversifyOverNeuralQuery() {
        String modelId = null;
        try {
            modelId = prepareModel();
            createPipelineProcessor(modelId, PIPELINE, ProcessorType.TEXT_EMBEDDING);
            // Index: text_knn is a top-level knn_vector populated by the pipeline from `text`.
            String indexConfig = buildIndexConfiguration();
            createIndexWithConfiguration(INDEX, indexConfig, PIPELINE);

            // Two near-duplicate texts and one distinct text; the embeddings cluster the near-dups together.
            ingestDocument(INDEX, "{\"" + TEXT_FIELD + "\":\"wireless bluetooth headphones\"}", "a");
            ingestDocument(INDEX, "{\"" + TEXT_FIELD + "\":\"bluetooth wireless headphone\"}", "b");
            ingestDocument(INDEX, "{\"" + TEXT_FIELD + "\":\"stainless steel water bottle\"}", "c");
            refresh(INDEX);

            // diversify over a standard leg whose query is `neural` on the knn_vector field.
            String body = "{\"retriever\":{\"diversify\":{\"vector_field\":\""
                + EMBEDDING_FIELD
                + "\",\"lambda\":0.1,\"window_size\":10,"
                + "\"retriever\":{\"standard\":{\"query\":{\"neural\":{\""
                + EMBEDDING_FIELD
                + "\":{\"query_text\":\"headphones\",\"model_id\":\""
                + modelId
                + "\",\"k\":10}}}}}}},\"size\":3}";

            List<String> ids = searchRetrieverIds(INDEX, body);
            assertEquals("diversify over neural returned 3 hits", 3, ids.size());
            // Under high diversity the distinct doc c is promoted, not left last (diversify changed the order
            // relative to the pure-relevance neural ranking where near-dups a,b dominate the top).
            assertNotEquals("distinct doc c promoted under high diversity", "c", ids.get(2));
        } finally {
            if (modelId != null) {
                deleteModel(modelId);
            }
            deleteIndex(INDEX);
        }
    }

    @SneakyThrows
    private void refresh(String index) {
        client().performRequest(new Request("POST", "/" + index + "/_refresh"));
    }

    @SneakyThrows
    @SuppressWarnings("unchecked")
    private List<String> searchRetrieverIds(String index, String body) {
        Request request = new Request("POST", "/" + index + "/_search");
        request.setJsonEntity(body);
        Response response = client().performRequest(request);
        Map<String, Object> map = createParser(
            org.opensearch.common.xcontent.json.JsonXContent.jsonXContent,
            EntityUtils.toString(response.getEntity())
        ).map();
        Map<String, Object> hits = (Map<String, Object>) map.get("hits");
        List<Map<String, Object>> hitList = (List<Map<String, Object>>) hits.get("hits");
        return hitList.stream().map(h -> (String) h.get("_id")).toList();
    }

    private String buildIndexConfiguration() {
        return "{\"settings\":{\"index.knn\":true,\"number_of_shards\":3,\"number_of_replicas\":0,"
            + "\"default_pipeline\":\""
            + PIPELINE
            + "\"},"
            + "\"mappings\":{\"properties\":{"
            + "\""
            + EMBEDDING_FIELD
            + "\":{\"type\":\"knn_vector\",\"dimension\":768,"
            + "\"method\":{\"name\":\"hnsw\",\"space_type\":\"l2\",\"engine\":\"lucene\"}},"
            + "\""
            + TEXT_FIELD
            + "\":{\"type\":\"text\"}}}}";
    }
}
