#pragma once

#include <cstddef>
#include <vector>



class Classifier
{
public:
	using Features = std::vector<float>;
	using Probas = std::vector<float>;

	virtual ~Classifier() = default;

	virtual size_t numClasses() const = 0;
	virtual size_t predict(const Features &) const = 0;
	virtual Probas predictProba(const Features &) const = 0;
};
