#include "graph.h"
#include <iostream>
#include <string>

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

	auto bfsVertices = graph.BFSCPU(start, end);

	for (int u : bfsVertices) {
		std::cout << u << " ";
	}

	return 0;
}