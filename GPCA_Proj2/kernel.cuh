#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "graph.h"
#include <stdio.h>
#include <iostream>
#include <string>

struct d_Ptrs1 {
	int* d_currentFrontier = nullptr;
	int* d_nextFrontier = nullptr;
	int* d_neighbours = nullptr;
	int* d_offsets = nullptr;
	int* d_parents = nullptr;
	int* d_nextFrontierSize = nullptr;
};

struct d_Ptrs2 {
	int* d_currentFrontier = nullptr;
	int* d_nextFrontier = nullptr;
	int* d_neighbours = nullptr;
	int* d_offsets = nullptr;
	int* d_parents = nullptr;
	int* d_nextFrontierSize = nullptr;
	int* d_frontierOffsets = nullptr;
	int* d_blockOffsets = nullptr;
	int* d_superBlockOffsets = nullptr;
};

__global__ void BFS1Kernel(int* d_neighbours, int* d_offsets, int* d_parents, int* d_currentFrontier, int* d_nextFrontier,
	int d_currentFrontierSize, int* d_nextFrontierSize);
__global__ void visitNeighboursKernel(int* d_neighbours, int* d_offsets, int* d_parents, int* d_currentFrontier, int d_currentFrontierSize, int* d_frontierOffsets);
__global__ void computeOffsetsKernel(int N, int* inArr, int* outArr);
__global__ void computeSuperBlockOffsetsKernel(int d_superBlocksCount, int* d_superBlockOffsets);
__global__ void makeNextFrontierKernel(int* d_neighbours, int* d_offsets, int* d_parents, int* d_currentFrontier, int* d_nextFrontier, int d_currentFrontierSize,
	int* d_frontierOffsets, int* d_blockOffsets, int* d_superBlockOffsets, int* nextFrontierSize);

std::list<int> BFSGPU1(Graph& graph, int start, int end, const char* outputFilename);
std::list<int> BFSGPU2(Graph& graph, int start, int end, const char* outputFilename);
int cudaCheckStatus1(cudaError_t cudaStatus, d_Ptrs1& dPtrs, const char* errorMessage, int* parents, int* neighbours, int* offsets);
void freeCudaMemory1(d_Ptrs1& dPtrs);
int cudaCheckStatus2(cudaError_t cudaStatus, d_Ptrs2& dPtrs, const char* errorMessage, int* parents, int* neighbours, int* offsets);
void freeCudaMemory2(d_Ptrs2& dPtrs);