#include "org_opensearch_neuralsearch_jni_NativeVsagService.h"
#include <vsag/vsag.h>

#include <iostream>
#include <fstream>

#include <string>
#include <vector>
#include <memory>

#include <jni.h>
#include "jni_util.h"
#include <unordered_set>

static neuralsearch_jni::JNIUtil jniUtil;
static const jint JNI_VERSION = JNI_VERSION_1_1;

jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    // Obtain the JNIEnv from the VM and confirm JNI_VERSION
    JNIEnv* env;
    if (vm->GetEnv((void**)&env, JNI_VERSION) != JNI_OK) {
        return JNI_ERR;
    }

    jniUtil.Initialize(env);

    return JNI_VERSION;
}

void JNI_OnUnload(JavaVM *vm, void *reserved) {
    JNIEnv* env;
    vm->GetEnv((void**)&env, JNI_VERSION);
    jniUtil.Uninitialize(env);
}

// Private helper to get the native vsag::Index* from a jlong
static vsag::Index* GetIndexPtr(jlong jIndexPtr) {
    auto sharedIndexPtr = reinterpret_cast<std::shared_ptr<vsag::Index>*>(jIndexPtr);
    if (!sharedIndexPtr) {
        throw std::runtime_error("Invalid native index pointer (nullptr)");
    }
    return sharedIndexPtr->get();
}

JNIEXPORT void JNICALL Java_org_opensearch_neuralsearch_jni_NativeVsagService_init
  (JNIEnv *, jobject) {
    vsag::init();
}


/*
 * Class:     org_opensearch_neuralsearch_jni_NativeVsagService
 * Method:    createIndex
 * Signature: (Ljava/lang/String;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_org_opensearch_neuralsearch_jni_NativeVsagService_createIndex
  (JNIEnv* env, jobject obj, jstring jIndexType, jstring jBuildParams) {
    try {
        std::string indexType(jniUtil.ConvertJavaStringToCppString(env, jIndexType));
        std::string buildParams(jniUtil.ConvertJavaStringToCppString(env, jBuildParams));

        auto result = vsag::Factory::CreateIndex(indexType, buildParams);

        if (!result) {
            throw std::runtime_error("Failed to create the index using vsag lib. Error: " + result.error().message);
        }

        // Store the shared_ptr<Index> on the heap and return its pointer as jlong
        std::shared_ptr<vsag::Index>* indexPtr = new std::shared_ptr<vsag::Index>(std::move(*result));
        return reinterpret_cast<jlong>(indexPtr);

    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
        return jlong(0);
    }
  }

/*
 * Class:     org_opensearch_neuralsearch_jni_NativeVsagService
 * Method:    add
 * Signature: (JLorg/opensearch/neuralsearch/jni/VsagDataset;)Z
 */
JNIEXPORT jlongArray JNICALL Java_org_opensearch_neuralsearch_jni_NativeVsagService_add
  (JNIEnv * env, jobject obj, jlong jIndexPtr, jobject jDataSet) {
     try {
        auto indexPtr = GetIndexPtr(jIndexPtr);
        auto datasetPtr = jniUtil.ConvertJavaDatasetToCppDatasetPtr(env, jDataSet);

        auto result = indexPtr->Add(datasetPtr);  // result is tl::expected<std::vector<int64_t>, vsag::Error>
        if (!result.has_value()) {
            // handle error returned by Add()
            const auto& err = result.error();
            std::string msg = "vsag::Index::Add failed: " + err.message; // or use your actual API
            std::cerr << msg << std::endl;
            throw std::runtime_error(msg);
        }

        // success: get the failed IDs
        const std::vector<int64_t>& failedIds = result.value();
        return jniUtil.ConvertCppLongArrayToJavaLongArray(env, failedIds);
     } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
        return nullptr;
     }
  }


/*
 * Class:     org_opensearch_neuralsearch_jni_NativeVsagService
 * Method:    knnSearch
 * Signature: (JLorg/opensearch/neuralsearch/jni/VsagSparseDataset;ILjava/lang/String;)[Lorg/opensearch/neuralsearch/jni/VsagSearchResult;
 */
JNIEXPORT jobjectArray JNICALL Java_org_opensearch_neuralsearch_jni_NativeVsagService_knnSearch
  (JNIEnv * env, jobject obj, jlong jIndexPtr, jobject jDataSet, jint jk, jstring jSearchParams) {
        try {
            auto indexPtr = GetIndexPtr(jIndexPtr);

            // Convert the Java query dataset to C++ dataset
            auto datasetPtr = jniUtil.ConvertJavaDatasetToCppDatasetPtr(env, jDataSet);

            // Convert search parameters
            int64_t k = static_cast<int64_t>(jk);
            std::string searchParams = jniUtil.ConvertJavaStringToCppString(env, jSearchParams);

            // Execute KNN search
            auto result = indexPtr->KnnSearch(datasetPtr, jk, searchParams);
            if (!result.has_value()) {
                const auto& err = result.error();
                std::string msg = "vsag::Index::KnnSearch failed: " + err.message;
                std::cerr << msg << std::endl;
                throw std::runtime_error(msg);
            }

            // Convert search results to Java array
            const std::shared_ptr<vsag::Dataset>& searchResults = result.value();
            return jniUtil.ConvertCppSearchResultsToJava(env, searchResults);

        } catch (...) {
            jniUtil.CatchCppExceptionAndThrowJava(env);
            return nullptr;
        }
  }

/*
* Class:     org_opensearch_neuralsearch_jni_NativeVsagService
* Method:    serializeIndex
* Signature: (JLjava/lang/String;)Z
*/
JNIEXPORT void JNICALL Java_org_opensearch_neuralsearch_jni_NativeVsagService_serializeIndex
(JNIEnv * env, jobject obj, jlong jIndexPtr, jstring jFilePath) {
   try {
        auto index = GetIndexPtr(jIndexPtr);
        std::string filePath = jniUtil.ConvertJavaStringToCppString(env, jFilePath);
        std::ofstream out_stream(filePath, std::ios::binary);
        if (!out_stream.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + filePath);
        }

        auto serialize_result = index->Serialize(out_stream);
        out_stream.close();

        if (!serialize_result.has_value()) {
            throw std::runtime_error("Index serialization failed: " + serialize_result.error().message);
        }

        std::cout << "Index serialized to " << filePath << std::endl;

   } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
   }
}

/*
* Class:     org_opensearch_neuralsearch_jni_NativeVsagService
* Method:    deserializeIndex
* Signature: (JLjava/lang/String;Ljava/lang/String;)Z
*/
JNIEXPORT void JNICALL Java_org_opensearch_neuralsearch_jni_NativeVsagService_deserializeIndex
(JNIEnv * env, jobject obj, jlong jIndexPtr, jstring jFilePath) {
    try {
        auto index = GetIndexPtr(jIndexPtr);
        std::string filePath = jniUtil.ConvertJavaStringToCppString(env, jFilePath);

        std::ifstream in_stream(filePath, std::ios::binary);
        if (!in_stream.is_open()) {
            throw std::runtime_error("Failed to open file for reading: " + filePath);
        }

        auto deserialize_result = index->Deserialize(in_stream);
        in_stream.close();

        if (!deserialize_result.has_value()) {
            throw std::runtime_error("Index deserialization failed: " + deserialize_result.error().message);
        }

        std::cout << "Index deserialized from " << filePath << std::endl;
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

/*
 * Class:     org_opensearch_neuralsearch_jni_NativeVsagService
 * Method:    cleanup
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_opensearch_neuralsearch_jni_NativeVsagService_cleanup
  (JNIEnv * env, jobject obj, jlong jIndexPtr) {
     try {
            if (jIndexPtr == 0) {
                // Nothing to clean
                 std::cout << "Native VSAG index nothing to clean." << std::endl;
                return;
            }

            // Cast back to shared_ptr pointer
            auto sharedIndexPtr = reinterpret_cast<std::shared_ptr<vsag::Index>*>(jIndexPtr);

            // Delete the shared_ptr to release ownership and decrease ref count
            delete sharedIndexPtr;

            // Optional: log cleanup
            std::cout << "Native VSAG index cleaned up." << std::endl;
     } catch (...) {
        // Catch C++ exceptions and throw to Java
        jniUtil.CatchCppExceptionAndThrowJava(env);
     }
  }