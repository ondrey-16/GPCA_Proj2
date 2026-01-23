#define _CRT_SECURE_NO_WARNINGS

#include "graph.h"
#include "kernel.cuh"
#include "pathTester.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <chrono>

std::list<int> BFSCPU(Graph& graph, int start, int end, const char* outputFilename);

int main(int argc, char** argv) {
	if (argc != 7) {
		std::cerr << "Wrong input arguments count\n";
		return 1;
	}

	const char* format = argv[1];
	if (strcmp(format, "txt_dir") != 0 && strcmp(format, "txt_undir") != 0) {
		std::cerr << "Wrong format type\n";
		return 1;
	}

	const char* method = argv[2];
	if (strcmp(method, "cpu") != 0 && strcmp(method, "gpu1") != 0 && strcmp(method, "gpu2") != 0) {
		std::cerr << "Wrong method type\n";
		return 1;
	}

	int start = atoi(argv[3]);
	int end = atoi(argv[4]);

	if (start < 0 || end < 0) {
		std::cerr << "Wrong input start and/or end vertex numbers\n";
		return 1;
	}

	const char* filename = argv[5];
	const char* outputFilename= argv[6];
	Graph graph;

	auto timeStart = std::chrono::high_resolution_clock::now();
	if (!graph.readGraphFromFile(filename, format)) {
		return 1;
	}
	auto timeEnd = std::chrono::high_resolution_clock::now();
	auto durationReadingFile = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart);
	std::cout << std::endl << "Reading file time: " << durationReadingFile.count() << "ms" << std::endl;

	std::cout << std::endl << "|V| = " << graph.N << ", |E| = " << graph.EdgeCount << std::endl;

	std::list<int> path;
	if (strcmp(method, "cpu") == 0) {
		path = BFSCPU(graph, start, end, outputFilename);
	}
	else if (strcmp(method, "gpu1") == 0) {
		path = BFSGPU1(graph, start, end, outputFilename);
	}
	else if (strcmp(method, "gpu2") == 0) {
		path = BFSGPU2(graph, start, end, outputFilename);
	}

	std::cout << std::endl << "Testing if path exists: ";
	std::string res = (pathTester(graph, path)) ? "Passed" : "Failed";
	std::cout << res << std::endl << "Path length: " << path.size();

	return 0;
}

std::list<int> BFSCPU(Graph& graph, int start, int end, const char* outputFilename) {
	std::cout << std::endl << "CPU: " << std::endl;
	auto timeStart = std::chrono::high_resolution_clock::now();

	auto bfsVertices = graph.BFSCPU(start, end);

	FILE* f = fopen(outputFilename, "w");

	auto timeEnd = std::chrono::high_resolution_clock::now();
	auto durationCPU = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart);
	std::cout << "Time on CPU: " << durationCPU.count() << " ms." << std::endl;

	for (int u : bfsVertices) {
		fprintf(f, "%d\n", u);
	}
	return bfsVertices;
}