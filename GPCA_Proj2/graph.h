#pragma once
#include <iostream>
#include <vector>
#include <string>

class Graph {
private:
	std::vector<std::vector<int>> graph;
	void pushEdge(int start, int end);

public:
	Graph() = default;
	bool readGraphFromFile(std::string filename);
	friend std::ostream& operator<<(std::ostream& out, const Graph& g);
};