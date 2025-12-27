#include "graph.h"
#include "kernel.cuh"
#include <iostream>
#include <string>
#include <chrono>

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

	const char* filename = argv[1];
	Graph graph;

	auto timeStart = std::chrono::high_resolution_clock::now();
	if (!graph.readGraphFromFile(filename)) {
		return 1;
	}
	auto timeEnd = std::chrono::high_resolution_clock::now();
	auto durationReadingFile = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart);
	std::cout << std::endl << "Reading file time: " << durationReadingFile.count() << "ms" << std::endl;

	timeStart = std::chrono::high_resolution_clock::now();
	BFSCPU(graph, start, end);
	timeEnd = std::chrono::high_resolution_clock::now();
	auto durationCPU = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart);

	timeStart = std::chrono::high_resolution_clock::now();
	BFSGPU1(graph, start, end);
	timeEnd = std::chrono::high_resolution_clock::now();
	auto durationGPU = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart);

	std::cout << std::endl << "Time on CPU: " << durationCPU.count() << "ms. Time on GPU: " << durationGPU.count() << "ms." << std::endl;

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