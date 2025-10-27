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

#ifndef OPENSEARCH_NEURALSEARCH_JNI_UTIL_H
#define OPENSEARCH_NEURALSEARCH_JNI_UTIL_H

#include <jni.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <vsag/vsag.h>

namespace neuralsearch_jni {

    // Interface for making calls to JNI
    struct JNIUtilInterface {
        // -------------------------- EXCEPTION HANDLING ----------------------------
        // Takes the name of a Java exception type and a message and throws the corresponding exception
        // to the JVM
        virtual void ThrowJavaException(JNIEnv* env, const char* type, const char* message) = 0;

        // Checks if an exception occurred in the JVM and if so throws a C++ exception
        // This should be called after some calls to JNI functions
        virtual void HasExceptionInStack(JNIEnv* env) = 0;

        // HasExceptionInStack with ability to specify message
        virtual void HasExceptionInStack(JNIEnv* env, const char *message) = 0;

        // Catches a C++ exception and throws the corresponding exception to the JVM
        virtual void CatchCppExceptionAndThrowJava(JNIEnv* env) = 0;
        // --------------------------------------------------------------------------

        // ------------------------------ JAVA FINDERS ------------------------------
        // Find a java class given a particular name
        virtual jclass FindClass(JNIEnv * env, const std::string& className) = 0;

        // Find a java method given a particular class, name and signature
        virtual jmethodID FindMethod(JNIEnv * env, const std::string& className, const std::string& methodName) = 0;

        // --------------------------------------------------------------------------

        // ------------------------- JAVA TO CPP CONVERTERS -------------------------
        // Returns cpp copied string from the Java string and releases the JNI Resource
        virtual std::string ConvertJavaStringToCppString(JNIEnv * env, jstring javaString) = 0;

        virtual vsag::DatasetPtr ConvertJavaDatasetToCppDatasetPtr(JNIEnv* env, jobject jDataset) = 0;

        // --------------------------------------------------------------------------

        // ------------------------- CPP TO JAVA CONVERTERS -------------------------
        virtual jlongArray ConvertCppLongArrayToJavaLongArray(JNIEnv * env, const std::vector<int64_t>& longArray) = 0;

        virtual jobjectArray ConvertCppSearchResultsToJava(JNIEnv* env, const std::shared_ptr<vsag::Dataset>& datasetPtr) = 0;
        // --------------------------------------------------------------------------



    };

    // Class that implements JNIUtilInterface methods
    class JNIUtil final : public JNIUtilInterface {
    public:
        // Initialize and Uninitialize methods are used for caching/cleaning up Java classes and methods
        void Initialize(JNIEnv* env);
        void Uninitialize(JNIEnv* env);

        void ThrowJavaException(JNIEnv* env, const char* type = "", const char* message = "") final;
        void HasExceptionInStack(JNIEnv* env) final;
        void HasExceptionInStack(JNIEnv* env, const char* message) final;
        void CatchCppExceptionAndThrowJava(JNIEnv* env) final;
        jclass FindClass(JNIEnv * env, const std::string& className) final;
        jmethodID FindMethod(JNIEnv * env, const std::string& className, const std::string& methodName) final;
        std::string ConvertJavaStringToCppString(JNIEnv * env, jstring javaString) final;
        vsag::DatasetPtr ConvertJavaDatasetToCppDatasetPtr(JNIEnv* env, jobject jDataset) final;
        jlongArray ConvertCppLongArrayToJavaLongArray(JNIEnv * env, const std::vector<int64_t>& longArray) final;
        jobjectArray ConvertCppSearchResultsToJava(JNIEnv* env, const std::shared_ptr<vsag::Dataset>& datasetPtr) final;

    private:
        std::unordered_map<std::string, jclass> cachedClasses;
        std::unordered_map<std::string, jmethodID> cachedMethods;
    };  // class JNIUtil
}
#endif // OPENSEARCH_NEURALSEARCH_JNI_UTIL_H
