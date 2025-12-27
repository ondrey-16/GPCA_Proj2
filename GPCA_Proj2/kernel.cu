#include "kernel.cuh"

__global__ void BFS1Kernel(int* d_neighbours, int* d_offsets, int* d_parents, int* d_currentFrontier, int* d_nextFrontier,
	int d_currentFrontierSize, int* d_nextFrontierSize) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= d_currentFrontierSize) {
		return;
	}

	int u = d_currentFrontier[tid];

	for (int i = d_offsets[u]; i < d_offsets[u + 1]; i++) {
		int v = d_neighbours[i];
		if (atomicCAS(&d_parents[v], -1, u) == -1) {
			int pos = atomicAdd(d_nextFrontierSize, 1);
			d_nextFrontier[pos] = v;
		}
	}
}

int BFSGPU1(Graph& graph, int start, int end) {
	d_Ptrs dPtrs;

	int frontierSize;
	int* parents = new int[graph.N];
	int* frontier = new int[graph.N];
	int* neighbours = new int[graph.EdgeCount];
	int* offsets = new int[graph.N + 1];
	std::copy(graph.neighbours.begin(), graph.neighbours.end(), neighbours);
	std::copy(graph.offsets.begin(), graph.offsets.end(), offsets);
	std::list<int> path;

	for (int i = 0; i < graph.N; i++) {
		parents[i] = -1;
	}
	parents[start] = -2;

	if (cudaCheckStatus(cudaMalloc((void**)&dPtrs.d_currentFrontier, sizeof(int) * graph.N), dPtrs, "cudaMalloc error")) {
		return 1;
	}
	if (cudaCheckStatus(cudaMalloc((void**)&dPtrs.d_nextFrontier, sizeof(int) * graph.N), dPtrs, "cudaMalloc error")) {
		return 1;
	}
	if (cudaCheckStatus(cudaMalloc((void**)&dPtrs.d_neighbours, sizeof(int) * graph.EdgeCount), dPtrs, "cudaMalloc error")) {
		return 1;
	}
	if (cudaCheckStatus(cudaMalloc((void**)&dPtrs.d_offsets, sizeof(int) * (graph.N + 1)), dPtrs, "cudaMalloc error")) {
		return 1;
	}
	if (cudaCheckStatus(cudaMalloc((void**)&dPtrs.d_parents, sizeof(int) * graph.N), dPtrs, "cudaMalloc error")) {
		return 1;
	}
	if (cudaCheckStatus(cudaMalloc((void**)&dPtrs.d_nextFrontierSize, sizeof(int)), dPtrs, "cudaMalloc error")) {
		return 1;
	}

	frontierSize = 1;
	frontier[0] = start;

	if (cudaCheckStatus(cudaMemcpy(dPtrs.d_neighbours, neighbours, sizeof(int) * graph.EdgeCount, cudaMemcpyHostToDevice), dPtrs, "cudaMemcpy error")) {
		return 1;
	}
	if (cudaCheckStatus(cudaMemcpy(dPtrs.d_offsets, offsets, sizeof(int) * (graph.N + 1), cudaMemcpyHostToDevice), dPtrs, "cudaMemcpy error")) {
		return 1;
	}
	if (cudaCheckStatus(cudaMemcpy(dPtrs.d_parents, parents, sizeof(int) * graph.N, cudaMemcpyHostToDevice), dPtrs, "cudaMemcpy error")) {
		return 1;
	}

	dim3 blockDim(256);

	while (frontierSize) {
		if (cudaCheckStatus(cudaMemcpy(dPtrs.d_currentFrontier, frontier, sizeof(int) * frontierSize, cudaMemcpyHostToDevice), dPtrs, "cudaMemcpy error")) {
			return 1;
		}
		if (cudaCheckStatus(cudaMemset(dPtrs.d_nextFrontierSize, 0, sizeof(int)), dPtrs, "cudaMemcpy error")) {
			return 1;
		}

		dim3 gridDim((frontierSize + blockDim.x - 1) / blockDim.x);
		BFS1Kernel << <gridDim, blockDim >> > (dPtrs.d_neighbours, dPtrs.d_offsets, dPtrs.d_parents, dPtrs.d_currentFrontier,
			dPtrs.d_nextFrontier, frontierSize, dPtrs.d_nextFrontierSize);

		auto cudaStatus = cudaDeviceSynchronize();
		if (cudaCheckStatus(cudaStatus, dPtrs, "BFS1Kernel error")) {
			std::cerr << cudaStatus;
			return 1;
		}

		if (cudaCheckStatus(cudaMemcpy(&frontierSize, dPtrs.d_nextFrontierSize, sizeof(int), cudaMemcpyDeviceToHost), dPtrs, "cudaMemcpy error")) {
			return 1;
		}
		if (cudaCheckStatus(cudaMemcpy(frontier, dPtrs.d_nextFrontier, sizeof(int) * frontierSize, cudaMemcpyDeviceToHost), dPtrs, "cudaMemcpy error")) {
			return 1;
		}
		if (cudaCheckStatus(cudaMemcpy(parents + end, dPtrs.d_parents + end, sizeof(int), cudaMemcpyDeviceToHost), dPtrs, "cudaMemcpy error")) {
			return 1;
		}

		if (parents[end] != -1) {
			if (cudaCheckStatus(cudaMemcpy(parents, dPtrs.d_parents, sizeof(int) * graph.N, cudaMemcpyDeviceToHost), dPtrs, "cudaMemcpy error")) {
				return 1;
			}
			break;
		}
	}

	if (parents[end] != -1) {
		int u = end;

		path.push_front(u);
		u = parents[u];
		while (u != -1 && u != -2) {
			path.push_front(u);
			u = parents[u];
		}
	}

	for (int u : path) {
		std::cout << u << " ";
	}

	freeCudaMemory(dPtrs);
	delete[] parents;
	delete[] frontier;
	delete[] neighbours;
	delete[] offsets;

	return 0;
}

int cudaCheckStatus(cudaError_t cudaStatus, d_Ptrs& dPtrs, const char* errorMessage) {
	if (cudaStatus != cudaSuccess) {
		std::cerr << errorMessage;
		freeCudaMemory(dPtrs);
		return 1;
	}
	return 0;
}

void freeCudaMemory(d_Ptrs& dPtrs) {
	cudaFree(dPtrs.d_currentFrontier);
	cudaFree(dPtrs.d_neighbours);
	cudaFree(dPtrs.d_nextFrontier);
	cudaFree(dPtrs.d_nextFrontierSize);
	cudaFree(dPtrs.d_offsets);
	cudaFree(dPtrs.d_parents);
}