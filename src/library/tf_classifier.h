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
	size_t num_classes() const override;
	size_t predict(const features_t &) const override;
	probas_t predict_proba(const features_t &) const override;

protected:
	static void delete_tf_session(TF_Session *);

	using tf_graph = std::unique_ptr<TF_Graph, decltype(&TF_DeleteGraph)>;
	using tf_buffer = std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)>;
	using tf_import_graph_def_options = std::unique_ptr<TF_ImportGraphDefOptions, decltype(&TF_DeleteImportGraphDefOptions)>;
	using tf_status = std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)>;
	using tf_session_options = std::unique_ptr<TF_SessionOptions, decltype(&TF_DeleteSessionOptions)>;
	using tf_tensor = std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)>;
	using tf_session = std::unique_ptr<TF_Session, decltype(&delete_tf_session)>;

protected:
	tf_graph _graph { TF_NewGraph(), TF_DeleteGraph };
	tf_session_options _session_opts { TF_NewSessionOptions(), TF_DeleteSessionOptions };
	tf_session _session = { nullptr, delete_tf_session };
	TF_Operation *_input_op { nullptr };
	TF_Operation *_output_op { nullptr };

	int _width;
	int _height;
};
