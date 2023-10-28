#include <filesystem>
#include <iostream>

#include <boost/format.hpp>

#include "tf_classifier.h"

const char *const USAGE_MESSAGE = "Usage: fashio_mnist <DATA> <MODEL>";
const char *const PATH_NOT_EXIST_MESSAGE = "Path '%s' not exist";

const size_t IMAGE_WIDTH = 28;
const size_t IMAGE_HEIGHT = 28;



int main(int argc, char *argv[])
{
	if (argc != 3)
	{
		std::cout << USAGE_MESSAGE << std::endl;
		return EXIT_FAILURE;
	}

	std::string dataPath = argv[1];
	std::string modelPath = argv[2];

	for (const auto &path : { dataPath, modelPath })
	{
		bool exist = std::filesystem::exists(path);
		if (!exist)
		{
			std::cout << USAGE_MESSAGE << std::endl;
			std::cout << (boost::format { PATH_NOT_EXIST_MESSAGE } % path) << std::endl;
			return EXIT_FAILURE;
		}
	}

	auto clf = TfClassifier { modelPath, IMAGE_WIDTH, IMAGE_HEIGHT };

	std::cout << "Hello, World!" << std::endl;
	return EXIT_SUCCESS;
}
