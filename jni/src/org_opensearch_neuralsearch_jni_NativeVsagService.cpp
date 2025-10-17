#include "org_opensearch_neuralsearch_jni_NativeVsagService.h"
#include <vsag/vsag.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>

JNIEXPORT void JNICALL Java_org_opensearch_neuralsearch_jni_NativeVsagService_init
  (JNIEnv *, jobject) {
    vsag::init();
}
