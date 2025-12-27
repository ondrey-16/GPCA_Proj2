#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "graph.h"
#include <stdio.h>
#include <iostream>
#include <string>

struct d_Ptrs {
	int* d_currentFrontier = nullptr;
	int* d_nextFrontier = nullptr;
	int* d_neighbours = nullptr;
	int* d_offsets = nullptr;
	int* d_parents = nullptr;
	int* d_nextFrontierSize = nullptr;
};

__global__ void BFS1Kernel(int* d_neighbours, int* d_offsets, int* d_parents, int* d_currentFrontier, int* d_nextFrontier,
	int d_currentFrontierSize, int* d_nextFrontierSize);

int BFSGPU1(Graph& graph, int start, int end);
int cudaCheckStatus(cudaError_t cudaStatus, d_Ptrs& dPtrs, const char* errorMessage);
void freeCudaMemory(d_Ptrs& dPtrs);