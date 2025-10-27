/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

 #include "jni_util.h"

 #include <jni.h>
 #include <new>
 #include <stdexcept>
 #include <string>
 #include <vector>
 #include <vsag/vsag.h>

 #include <iostream>

 namespace neuralsearch_jni {
    void JNIUtil::Initialize(JNIEnv *env) {
        // Followed recommendation from this SO post: https://stackoverflow.com/a/13940735
        jclass tempLocalClassRef;

        tempLocalClassRef = env->FindClass("java/io/IOException");
        this->cachedClasses["java/io/IOException"] = (jclass) env->NewGlobalRef(tempLocalClassRef);
        env->DeleteLocalRef(tempLocalClassRef);

        tempLocalClassRef = env->FindClass("java/lang/Exception");
        this->cachedClasses["java/lang/Exception"] = (jclass) env->NewGlobalRef(tempLocalClassRef);
        env->DeleteLocalRef(tempLocalClassRef);

        // Cache VsagSparseVector and its methods
        tempLocalClassRef = env->FindClass("org/opensearch/neuralsearch/jni/VsagSparseVector");
        this->cachedClasses["org/opensearch/neuralsearch/jni/VsagSparseVector"] = (jclass) env->NewGlobalRef(tempLocalClassRef);
        this->cachedMethods["org/opensearch/neuralsearch/jni/VsagSparseVector:getLength"] = env->GetMethodID(tempLocalClassRef, "getLength", "()I");
        this->cachedMethods["org/opensearch/neuralsearch/jni/VsagSparseVector:getIds"] = env->GetMethodID(tempLocalClassRef, "getIds", "()[I");
        this->cachedMethods["org/opensearch/neuralsearch/jni/VsagSparseVector:getValues"] = env->GetMethodID(tempLocalClassRef, "getValues", "()[F");
        this->cachedMethods["org/opensearch/neuralsearch/jni/VsagSparseVector:getDocId"] = env->GetMethodID(tempLocalClassRef, "getDocId", "()J");
        env->DeleteLocalRef(tempLocalClassRef);

        tempLocalClassRef = env->FindClass("org/opensearch/neuralsearch/jni/VsagSearchResult");
        this->cachedClasses["org/opensearch/neuralsearch/jni/VsagSearchResult"] = (jclass) env->NewGlobalRef(tempLocalClassRef);
        this->cachedMethods["org/opensearch/neuralsearch/jni/VsagSearchResult:<init>"] = env->GetMethodID(tempLocalClassRef, "<init>", "(JF)V");
        env->DeleteLocalRef(tempLocalClassRef);

        // Cache VsagDataset and its getSparseVectors method
        tempLocalClassRef = env->FindClass("org/opensearch/neuralsearch/jni/VsagDataset");
        this->cachedClasses["org/opensearch/neuralsearch/jni/VsagDataset"] = (jclass) env->NewGlobalRef(tempLocalClassRef);
        this->cachedMethods["org/opensearch/neuralsearch/jni/VsagDataset:getSparseVectors"] = env->GetMethodID(tempLocalClassRef, "getSparseVectors", "()[Lorg/opensearch/neuralsearch/jni/VsagSparseVector;");
        env->DeleteLocalRef(tempLocalClassRef);
    }

    void JNIUtil::Uninitialize(JNIEnv* env) {
        // Delete all classes that are now global refs
        for (auto & cachedClasse : this->cachedClasses) {
            env->DeleteGlobalRef(cachedClasse.second);
        }
        this->cachedClasses.clear();
        this->cachedMethods.clear();
    }

    void JNIUtil::ThrowJavaException(JNIEnv* env, const char* type, const char* message) {
        jclass newExcCls = env->FindClass(type);
        if (newExcCls != nullptr) {
            env->ThrowNew(newExcCls, message);
            env->DeleteLocalRef(newExcCls);
        }
        // If newExcCls isn't found, NoClassDefFoundError will be thrown
    }

    void JNIUtil::HasExceptionInStack(JNIEnv* env) {
        this->HasExceptionInStack(env, "Exception in jni occurred");
    }

    void JNIUtil::HasExceptionInStack(JNIEnv* env, const char* message) {
        if (env->ExceptionCheck() == JNI_TRUE) {
            throw std::runtime_error(message);
        }
    }

    void JNIUtil::CatchCppExceptionAndThrowJava(JNIEnv* env)
    {
        try {
            throw;
        }
        catch (const std::bad_alloc& rhs) {
            this->ThrowJavaException(env, "java/io/IOException", rhs.what());
        }
        catch (const std::runtime_error& re) {
            this->ThrowJavaException(env, "java/lang/Exception", re.what());
        }
        catch (const std::exception& e) {
            this->ThrowJavaException(env, "java/lang/Exception", e.what());
        }
        catch (...) {
            this->ThrowJavaException(env, "java/lang/Exception", "Unknown exception occurred");
        }
    }

    jclass JNIUtil::FindClass(JNIEnv * env, const std::string& className) {
        if (this->cachedClasses.find(className) == this->cachedClasses.end()) {
            throw std::runtime_error("Unable to load class \"" + className + "\"");
        }

        return this->cachedClasses[className];
    }

    jmethodID JNIUtil::FindMethod(JNIEnv * env, const std::string& className, const std::string& methodName) {
        std::string key = className + ":" + methodName;
        if (this->cachedMethods.find(key) == this->cachedMethods.end()) {
            throw std::runtime_error("Unable to find \"" + methodName + "\" method");
        }

        return this->cachedMethods[key];
    }

    std::string JNIUtil::ConvertJavaStringToCppString(JNIEnv * env, jstring javaString) {
        if (javaString == nullptr) {
            throw std::runtime_error("String cannot be null");
        }

        const char *cString = env->GetStringUTFChars(javaString, nullptr);
        if (cString == nullptr) {
            this->HasExceptionInStack(env, "Unable to convert java string to cpp string");

            // Will only reach here if there is no exception in the stack, but the call failed
            throw std::runtime_error("Unable to convert java string to cpp string");
        }
        std::string cppString(cString);
        env->ReleaseStringUTFChars(javaString, cString);
        return cppString;
    }

    vsag::DatasetPtr JNIUtil::ConvertJavaDatasetToCppDatasetPtr(JNIEnv* env, jobject jDataset) {
        // Get VsagDataset class
        jclass datasetClass = this->FindClass(env, "org/opensearch/neuralsearch/jni/VsagDataset");
        jmethodID getSparseVectors = this->FindMethod(env, "org/opensearch/neuralsearch/jni/VsagDataset", "getSparseVectors");

        jobjectArray jSparseVectorsArray = (jobjectArray)env->CallObjectMethod(jDataset, getSparseVectors);
        jsize numElements = env->GetArrayLength(jSparseVectorsArray);

        auto svVec = new vsag::SparseVector[numElements];
        auto docIds = new int64_t[numElements];

        // Get VsagSparseVector class
        jclass svClass = this->FindClass(env, "org/opensearch/neuralsearch/jni/VsagSparseVector");
        jmethodID getLength = this->FindMethod(env, "org/opensearch/neuralsearch/jni/VsagSparseVector", "getLength");
        jmethodID getIds = this->FindMethod(env, "org/opensearch/neuralsearch/jni/VsagSparseVector", "getIds");
        jmethodID getValues = this->FindMethod(env, "org/opensearch/neuralsearch/jni/VsagSparseVector", "getValues");
        jmethodID getDocId = this->FindMethod(env, "org/opensearch/neuralsearch/jni/VsagSparseVector", "getDocId");

        for (jsize i = 0; i < numElements; ++i) {
            jobject jSV = env->GetObjectArrayElement(jSparseVectorsArray, i);

            jint len = env->CallIntMethod(jSV, getLength);
            jintArray jIds = (jintArray)env->CallObjectMethod(jSV, getIds);
            jfloatArray jValues = (jfloatArray)env->CallObjectMethod(jSV, getValues);
            jlong jDocId = (jlong)env->CallObjectMethod(jSV, getDocId);

            svVec[i].len_ = len;
            svVec[i].ids_ = new uint32_t[len];
            svVec[i].vals_ = new float[len];
            docIds[i] = jDocId;

            jint* idsElems = env->GetIntArrayElements(jIds, nullptr);
            jfloat* valsElems = env->GetFloatArrayElements(jValues, nullptr);

            for (int j = 0; j < len; ++j) {
                svVec[i].ids_[j] = static_cast<uint32_t>(idsElems[j]);
                svVec[i].vals_[j] = valsElems[j];
            }

            env->ReleaseIntArrayElements(jIds, idsElems, JNI_ABORT);
            env->ReleaseFloatArrayElements(jValues, valsElems, JNI_ABORT);

            env->DeleteLocalRef(jSV);
            env->DeleteLocalRef(jIds);
            env->DeleteLocalRef(jValues);
        }

        // Create C++ Dataset
        auto datasetPtr = vsag::Dataset::Make();
        datasetPtr->NumElements(numElements)->SparseVectors(svVec)->Ids(docIds)->Owner(true);
        return datasetPtr;
    }

    jlongArray JNIUtil::ConvertCppLongArrayToJavaLongArray(JNIEnv* env, const std::vector<int64_t>& longArray) {
        jlongArray javaLongArray = env->NewLongArray(longArray.size());
        if (javaLongArray == nullptr) {
            throw std::bad_alloc();
        }

        if (!longArray.empty()) {
            env->SetLongArrayRegion(javaLongArray, 0, longArray.size(), longArray.data());
        }

        return javaLongArray;
    }

    jobjectArray JNIUtil::ConvertCppSearchResultsToJava(JNIEnv* env, const std::shared_ptr<vsag::Dataset>& datasetPtr) {
        // Find VsagSearchResult class
        jclass resultCls = this->FindClass(env, "org/opensearch/neuralsearch/jni/VsagSearchResult");
        // Get constructor (adjust signature if your Java class differs)
        // Example: VsagSearchResult(long id, float score)
        jmethodID ctor = this->FindMethod(env, "org/opensearch/neuralsearch/jni/VsagSearchResult", "<init>");

        // Retrieve pointers to IDs and distances(scores)
        const int64_t* ids = datasetPtr->GetIds();
        const float* distances = datasetPtr->GetDistances();
        // Determine number of results from dataset
        jsize numResults = static_cast<jsize>(datasetPtr->GetDim());

        // Create Java array
        jobjectArray jArray = env->NewObjectArray(numResults, resultCls, nullptr);
        if (jArray == nullptr) {
            throw std::runtime_error("Failed to allocate VsagSearchResult array");
        }

        // Fill array
        for (jsize i = 0; i < numResults; ++i) {
            // not sure why the lib only stores the distance which is the 1 - inner_product so we need to convert it back as the score
            jobject jRes = env->NewObject(resultCls, ctor, static_cast<jlong>(ids[i]), 1 - static_cast<jfloat>(distances[i]));
            env->SetObjectArrayElement(jArray, i, jRes);
        }

        return jArray;

    }
 } // namespace neuralsearch_jni