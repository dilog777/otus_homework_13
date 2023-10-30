#include <filesystem>
#include <iostream>
#include <memory>

#include <boost/format.hpp>

#include "DataReader.h"
#include "ImageClassifier.h"

const char *const USAGE_MESSAGE = "Usage: fashio_mnist <DATA> <MODEL>";
const char *const PATH_NOT_EXIST_MESSAGE = "Path '%1%' not exist";
const char *const FILE_NOT_OPENED = "File '%1%' not opened";
const char *const ERROR_IN_LINE = "Error in line %1%: %2%";

const char *const INCORRECT_NUMBER_ARGUMENTS = "Incorrect number of arguments";
const char *const NO_DATA = "No data to evaluate the model";

const size_t IMAGE_WIDTH = 28;
const size_t IMAGE_HEIGHT = 28;
const size_t IMAGE_CLASS_COUNT = 10;



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

	DataReader reader;
	if (!reader.open(dataPath))
	{
		std::cout << (boost::format { FILE_NOT_OPENED } % dataPath) << std::endl;
		return EXIT_FAILURE;
	}

	std::unique_ptr<ImageClassifier> classifier;
	try
	{
		classifier = std::make_unique<ImageClassifier>(modelPath, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CLASS_COUNT);
	}
	catch (const std::exception &ex)
	{
		std::cout << ex.what() << std::endl;
		return EXIT_FAILURE;
	}

	int line = 0;
	int truePredictions = 0;
	int totalPredictions = 0;
	while (!reader.eof())
	{
		++line;

		std::string error;
		auto data = reader.readLine(error);
		if (!error.empty())
		{
			std::cout << (boost::format { ERROR_IN_LINE } % line % error) << std::endl;
			continue;
		}

		if (data.empty())
			continue;

		if (data.size() != IMAGE_WIDTH * IMAGE_HEIGHT + 1)
		{
			std::cout << (boost::format { ERROR_IN_LINE } % line % INCORRECT_NUMBER_ARGUMENTS) << std::endl;
			continue;
		}

		size_t predict;
		size_t expectedPredict = static_cast<int>(data.front());
		ImageClassifier::Features features { std::next(data.begin()), data.end() };

		try
		{
			predict = classifier->predict(features);
		}
		catch (const std::exception &ex)
		{
			std::cout << (boost::format { ERROR_IN_LINE } % line % ex.what()) << std::endl;
			continue;
		}

		if (expectedPredict == predict)
			++truePredictions;

		++totalPredictions;
	}

	if (totalPredictions == 0)
	{
		std::cout << NO_DATA << std::endl;
		return EXIT_FAILURE;
	}

	float accuracy = static_cast<float>(truePredictions) / totalPredictions;
	std::cout << accuracy << std::endl;

	return EXIT_SUCCESS;
}
