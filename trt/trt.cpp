#include "NvInfer.h"
#include <iostream>
#include "NvOnnxParser.h"

using namespace nvinfer1;
using namespace nvonnxparser;

int main (int argc, char** args) {
    class Logger : public ILogger {
        void log(Severity severity, const char* msg) override
        {
            // suppress info-level messages
            if (severity <= Severity::kWARNING)
                std::cout << msg << std::endl;
        }
    } logger;

    IBuilder* builder = createInferBuilder(logger);

    uint32_t flag = 1U <<static_cast<uint32_t>
    (NetworkDefinitionCreationFlag::kEXPLICIT_BATCH) 

    INetworkDefinition* network = builder->createNetworkV2(flag);

    IParser*  parser = createParser(*network, logger);

    parser->parseFromFile(modelFile, ILogger::Severity::kWARNING);
    for (int32_t i = 0; i < parser.getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }
}