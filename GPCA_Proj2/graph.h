#pragma once
#include <iostream>
#include <vector>
#include <list>
#include <cstring>

/// <summary>
/// Klasa przechowuj¹ca reprezentacjê grafu, wczytywanego z zadanego pliku w postaci skierowanej lub nieskierowanej.
/// </summary>
class Graph {
private:
	std::vector<std::vector<int>> graph;
	void pushEdge(int start, int end, const char* format);
	void transformGraph();
public:
	Graph() = default;
	~Graph() = default;
	int N = 0;
	int EdgeCount = 0;
	std::vector<int> neighbours;
	std::vector<int> offsets;

	bool readGraphFromFile(const char* filename, const char* format);
	std::list<int> BFSCPU(int start, int end);
	friend std::ostream& operator<<(std::ostream& out, const Graph& g);
};