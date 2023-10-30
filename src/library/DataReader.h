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

	bool endFile() const;
	std::vector<int> readLine(std::string &error);

private:
	std::ifstream _fileStream;
};
