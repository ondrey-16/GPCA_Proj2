#include "graph.h"
#include "kernel.cuh"
#include <iostream>
#include <string>

int BFSCPU(Graph& graph, int start, int end);

int main(int argc, char** argv) {
	if (argc != 4) {
		std::cerr << "Wrong input arguments count\n";
		return 1;
	}

	int start = atoi(argv[2]);
	int end = atoi(argv[3]);

	if (start < 0 || end < 0) {
		std::cerr << "Wrong input start and end vertices numbers\n";
		return 1;
	}

	std::string filename = argv[1];
	Graph graph;

	if (!graph.readGraphFromFile(filename)) {
		return 1;
	}

	BFSCPU(graph, start, end);
	BFSGPU1(graph, start, end);

	return 0;
}

int BFSCPU(Graph& graph, int start, int end) {
	auto bfsVertices = graph.BFSCPU(start, end);

	for (int u : bfsVertices) {
		std::cout << u << " ";
	}
	std::cout << std::endl;
	return 0;
}