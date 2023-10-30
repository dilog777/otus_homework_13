#include "DataReader.h"

const char *const VALUES_DELIMITER = ",";
const char *const ERROR_GETLINE = "Read line error";
const char *const ERROR_PARSE_LINE = "Parse line error";



std::vector<std::string> split(const std::string &str, const std::string &delimiter)
{
	std::vector<std::string> result;

	size_t posStart = 0;
	size_t posEnd = 0;
	while ((posEnd = str.find(delimiter, posStart)) != std::string::npos)
	{
		auto token = str.substr(posStart, posEnd - posStart);
		posStart = posEnd + delimiter.length();
		result.push_back(token);
	}

	result.push_back(str.substr(posStart));
	return result;
}



DataReader::~DataReader()
{
	close();
}



bool DataReader::open(const std::string &filePath)
{
	_fileStream.open(filePath);
	return _fileStream.is_open();
}



void DataReader::close()
{
	_fileStream.close();
}



bool DataReader::endFile() const
{
	return _fileStream.eof();
}



std::vector<int> DataReader::readLine(std::string &error)
{
	std::string line;
	if (!std::getline(_fileStream, line))
	{
		error = ERROR_GETLINE;
		return {};
	}

	auto strings = split(line,VALUES_DELIMITER);
	
	std::vector<int> result;
	try
	{
		for (const auto &str : strings)
			result.push_back(std::stoi(str));
	}
	catch (...)
	{
		error = ERROR_PARSE_LINE;
		return {};
	}

	return result;
}
