#include "graph.h"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
	if (argc != 2) {
		std::cerr << "Wrong input arguments\n";
	}

	std::string filename = argv[1];
	Graph graph;

	if (!graph.readGraphFromFile(filename)) {
		return 1;
	}

	std::cout << graph;

	return 0;
}