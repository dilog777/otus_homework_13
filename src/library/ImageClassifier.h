#pragma once

#include <memory>
#include <string>
#include <vector>

#include <tensorflow/c/c_api.h>



class ImageClassifier
{
public:
	using Features = std::vector<float>;
	using Probas = std::vector<float>;

	ImageClassifier(const std::string &modelPath, int width, int height, int classCount);

	ImageClassifier(const ImageClassifier &) = delete;
	ImageClassifier &operator=(const ImageClassifier &) = delete;

	size_t predict(const Features &features) const;
	Probas predictProba(const Features &features) const;

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

	const int _width;
	const int _height;
	const int _classCount;
};
