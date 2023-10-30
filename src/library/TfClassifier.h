#pragma once

#include <memory>
#include <string>

#include <tensorflow/c/c_api.h>

#include "classifier.h"



class TfClassifier : public Classifier
{
public:
	TfClassifier(const std::string &modelpath, const int width, const int height);

	TfClassifier(const TfClassifier &) = delete;
	TfClassifier &operator=(const TfClassifier &) = delete;

	// Classifier interface
	size_t numClasses() const override;
	size_t predict(const Features &) const override;
	Probas predictProba(const Features &) const override;

private:
	static void deleteTfSession(TF_Session *);

	using GraphPtr = std::unique_ptr<TF_Graph, decltype(&TF_DeleteGraph)>;
	using BufferPtr = std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)>;
	using ImportGraphDefOptionsPtr = std::unique_ptr<TF_ImportGraphDefOptions, decltype(&TF_DeleteImportGraphDefOptions)>;
	using StatusPtr = std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)>;
	using SessionOptionsPtr = std::unique_ptr<TF_SessionOptions, decltype(&TF_DeleteSessionOptions)>;
	using TensorPtr = std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)>;
	using SessionPtr = std::unique_ptr<TF_Session, decltype(&deleteTfSession)>;

	GraphPtr _graph { TF_NewGraph(), TF_DeleteGraph };
	SessionOptionsPtr _sessionOptions { TF_NewSessionOptions(), TF_DeleteSessionOptions };
	SessionPtr _session = { nullptr, deleteTfSession };
	TF_Operation *_inputOperation { nullptr };
	TF_Operation *_outputOperation { nullptr };

	int _width;
	int _height;
};
