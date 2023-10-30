#pragma once

#include <fstream>
#include <string>
#include <vector>



class DataReader
{
public:
	DataReader() = default;
	~DataReader();

	bool open(const std::string &filePath);
	void close();

	bool eof() const;
	std::vector<float> readLine(std::string &error);

private:
	std::ifstream _fileStream;
};
