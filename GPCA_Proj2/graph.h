#pragma once
#include <iostream>
#include <vector>
#include <list>
#include <string>

class Graph {
private:
	std::vector<std::vector<int>> graph;
	void pushEdge(int start, int end);
	void transformGraph();
public:
	Graph() = default;
	~Graph() = default;
	int N = 0;
	int EdgeCount = 0;
	std::vector<int> neighbours;
	std::vector<int> offsets;

	bool readGraphFromFile(std::string filename);
	std::list<int> BFSCPU(int start, int end);
	friend std::ostream& operator<<(std::ostream& out, const Graph& g);
};