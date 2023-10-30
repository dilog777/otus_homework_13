#include "ImageClassifier.h"

#include <algorithm>
#include <sstream>

const char *const SESSION_TAGS = "serve";
const int SESSION_TAGS_COUNT = 1;
const char *const INPUT_OPERATION_NAME = "serving_default_input";
const char *const OUTPUT_OPERATION_NAME = "StatefulPartitionedCall";



static void dummyDeleter([[maybe_unused]] void *data, [[maybe_unused]] size_t length, [[maybe_unused]] void *arg)
{
}



ImageClassifier::ImageClassifier(const std::string &modelPath, int width, int height, int classCount)
	: _width { width }
	, _height { height }
	, _classCount { classCount }
{
	StatusPtr status { TF_NewStatus(), TF_DeleteStatus };
	_session.reset(TF_LoadSessionFromSavedModel(_sessionOptions.get(), nullptr, modelPath.c_str(), &SESSION_TAGS, SESSION_TAGS_COUNT, _graph.get(), nullptr, status.get()));
	if (TF_GetCode(status.get()) != TF_OK)
	{
		std::stringstream ss;
		ss << " Unable to import graph from '" << modelPath << "': " << TF_Message(status.get());
		throw std::invalid_argument { ss.str() };
	}

	_inputOperation = TF_GraphOperationByName(_graph.get(), INPUT_OPERATION_NAME);
	if (!_inputOperation)
	{
		throw std::runtime_error { "Input not found" };
	}

	_outputOperation = TF_GraphOperationByName(_graph.get(), OUTPUT_OPERATION_NAME);
	if (!_inputOperation)
	{
		throw std::runtime_error { "Output not found" };
	}
}



size_t ImageClassifier::predict(const Features &features) const
{
	auto proba = predictProba(features);
	auto argmax = std::max_element(proba.begin(), proba.end());
	return std::distance(proba.begin(), argmax);
}



ImageClassifier::Probas ImageClassifier::predictProba(const Features &features) const
{
	// Preprocess input features
	Features preprocFeatures;
	preprocFeatures.reserve(features.size());

	// Divide each bytes by 255
	std::transform(features.begin(), features.end(), std::back_inserter(preprocFeatures), [](float val) { return val / 255; });
	std::vector<TF_Output> inputs;
	std::vector<TF_Tensor *> inputValues;

	TF_Output inputOpout = { _inputOperation, 0 };
	inputs.push_back(inputOpout);

	// Create variables to store the size of the input and output variables
	const int numBytesIn = _width * _height * sizeof(float);
	const int numBytesOut = _classCount * sizeof(float);

	// Set input dimensions - this should match the dimensionality of the input in
	// the loaded graph, in this case it's three dimensional.
	int64_t inputDims[] = { 1, _width, _height, 1 };
	int64_t outputDims[] = { 1, static_cast<int64_t>(_classCount) };

	TensorPtr input { TF_NewTensor(TF_FLOAT, inputDims, 4, reinterpret_cast<void *>(preprocFeatures.data()), numBytesIn, &dummyDeleter, 0), TF_DeleteTensor };
	inputValues.push_back(input.get());

	std::vector<TF_Output> outputs;
	TF_Output outputOpout = { _outputOperation, 0 };
	outputs.push_back(outputOpout);

	// Create TF_Tensor* vector
	std::vector<TF_Tensor *> outputValues(outputs.size(), nullptr);

	// Similar to creating the input tensor, however here we don't yet have the
	// output values, so we use TF_AllocateTensor()
	TensorPtr outputValue { TF_AllocateTensor(TF_FLOAT, outputDims, 2, numBytesOut), TF_DeleteTensor };
	outputValues.push_back(outputValue.get());

	StatusPtr status { TF_NewStatus(), TF_DeleteStatus };

	TF_SessionRun(_session.get(), nullptr, &inputs[0], &inputValues[0], static_cast<int>(inputs.size()), &outputs[0], &outputValues[0], static_cast<int>(outputs.size()), nullptr, 0, nullptr,
		status.get());
	if (TF_GetCode(status.get()) != TF_OK)
	{
		std::stringstream ss;
		ss << "Unable to run session from graph: " << TF_Message(status.get());
		throw std::runtime_error { ss.str() };
	}

	Probas probas;
	float *outVals = static_cast<float *>(TF_TensorData(outputValues[0]));
	for (int i = 0; i < _classCount; ++i)
		probas.push_back(*outVals++);

	return probas;
}



void ImageClassifier::deleteTfSession(TF_Session *SessionPtr)
{
	StatusPtr status { TF_NewStatus(), TF_DeleteStatus };
	TF_DeleteSession(SessionPtr, status.get());
	if (TF_GetCode(status.get()) != TF_OK)
	{
		std::stringstream ss;
		ss << " Unable to delete TF_Session: " << TF_Message(status.get());
		throw std::runtime_error { ss.str() };
	}
}
