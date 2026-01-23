#include "kernel.cuh"
#include <chrono>

/// <summary>
/// Funkcja kernela do pierwszego algorytmu działającego na GPU. Oparta jest na użyciu funkcji atomowych zapewniających bezpieczne markowanie rodziców 
/// odwiedzanych wierzchołków oraz odpowiednie inkrementowanie rozmiaru następnej kolejki. Jeden warp odpowiada jednemu wierzchołkowi - każdy wątek w warpie rozpatruje odrębną część sąsiadów.
/// </summary>
/// <param name="d_neighbours">Tablica sąsiadów wszystkich wierzchołków grafu dla reprezentacji CSR</param>
/// <param name="d_offsets">Tablica początków fragmentów tablicy sąsiadów wszystkich wierzchołków grafu dla reprezentacji CSR</param>
/// <param name="d_parents">Tablica służąca do oznaczania wierzchołków, z których bezpośrednio się doszło do tych rozpatrywanych</param>
/// <param name="d_currentFrontier">Kolejka wierzchołków dla danej iteracji</param>
/// <param name="d_nextFrontier">Kolejka zawierające wierzchołki, które będą rozpatrywnae w następnej iteracji</param>
/// <param name="d_currentFrontierSize">Rozmiar obecnej kolejki</param>
/// <param name="d_nextFrontierSize">Rozmiar następnej kolejki</param>
__global__ void BFS1Kernel(int* d_neighbours, int* d_offsets, int* d_parents, int* d_currentFrontier, int* d_nextFrontier,
	int d_currentFrontierSize, int* d_nextFrontierSize) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int wid = tid / 32;
	if (wid >= d_currentFrontierSize) {
		return;
	}
	int warpStride = tid % 32;

	int u = d_currentFrontier[wid];
	int start = d_offsets[u];
	int end = d_offsets[u + 1];
	for (int i = start + warpStride; i < end; i += 32) {
		int v = d_neighbours[i];
		if (atomicCAS(&d_parents[v], -1, u) == -1) {
			int pos = atomicAdd(d_nextFrontierSize, 1);
			d_nextFrontier[pos] = v;
		}
	}
}

/// <summary>
/// Funkcja kernela dla drugiego algorytmu BFS działającego na GPU. Podobnie jak w funkcji dla BFSGPU1 tutaj również jeden warp odpowiada jednemu wierzchołkowi a 
/// wątki dzielą się sąsiadami do rozpatrzenia. Przy użyciu funkcji atomicCAS markują odpowiednio pola tablicy d_parents. Wątki zliczają sumy nowo odwiedzonych wierzchołków, które za pomocą funkcji
/// __shfl_down_sync są odpowiednio sumowane, aby wydobyć liczbę wszystkich nowo odwiedzonych wierzchołków dla danego warpa.
/// </summary>
/// <param name="d_neighbours">Tablica sąsiadów wszystkich wierzchołków grafu dla reprezentacji CSR</param>
/// <param name="d_offsets">Tablica początków fragmentów tablicy sąsiadów wszystkich wierzchołków grafu dla reprezentacji CSR</param>
/// <param name="d_parents">Tablica służąca do oznaczania wierzchołków, z których bezpośrednio się doszło do tych rozpatrywanych</param>
/// <param name="d_currentFrontier">Kolejka wierzchołków dla danej iteracji</param>
/// <param name="d_nextFrontier">Kolejka zawierające wierzchołki, które będą rozpatrywnae w następnej iteracji</param>
/// <param name="d_currentFrontierSize">Rozmiar obecnej kolejki</param>
/// <param name="d_frontierOffsets">Tablica, do której zapisywane będą przesunięcia zapisów nowych wierzchołków do następnej kolejki w obrębie 1 bloku</param>
__global__ void visitNeighboursKernel(int* d_neighbours, int* d_offsets, int* d_parents, int* d_currentFrontier, int d_currentFrontierSize, int* d_frontierOffsets)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int wid = tid / 32; 
	if (wid >= d_currentFrontierSize) { 
		return; 
	} 
	int warpStride = tid % 32;

	int u = d_currentFrontier[wid];
	int start = d_offsets[u];
	int end = d_offsets[u + 1];
	int threadCount = 0;
	for (int i = start + warpStride; i < end; i += 32) {
		int v = d_neighbours[i];
		if (atomicCAS(&d_parents[v], -1, u) == -1) {
			threadCount++;
		}
	}
	for (int i = 16; i > 0; i /= 2) { 
		threadCount += __shfl_down_sync(0xFFFFFFFF, threadCount, i);
	} 
	if (!warpStride) { 
		d_frontierOffsets[wid] = threadCount;
	}
}

/// <summary>
/// Funkcja kernela dla drugiego algorytmu BFS działającego na GPU. Oblicza sumy prefiksowe w obrębie jednego bloku. Ponieważ w architekturze algorytmów ta funkcja jest wywoływana na 
/// blokach o rozmiarze 1024, dokonywane jest dwuetapowe zliczanie sum prefiksowych - najpierw w obrębie 1 warpu, a następnie obliczane są sumy prefiksowe dla każdego z tych warpów w obrębie bloku 
/// w analogiczny sposób, aby następnie dla każdego wątka można było dodać sumę prefiksową wewnątrz swojego warpu wraz z sumą prefiksową odpowiadającego mu wątku, co daje sumę prefiksową wątka
/// w obrębie całego bloku. Do tablicy wyjściowej dla każdego bloku przekazywana jest suma wszystkich wejściowych składników tego bloku.
/// </summary>
/// <param name="N">Rozmiar tablicy wejściowej</param>
/// <param name="inArr">Tablica wejściowa</param>
/// <param name="outArr">Tablica wyjściowa</param>
__global__ void computeOffsetsKernel(int N, int* inArr, int* outArr)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int warpStride = threadIdx.x % 32;
	int offset = 0;
	if (tid < N) {
		offset = inArr[tid];
	}
	for (int i = 1; i < 32; i *= 2) {
		int tmp = __shfl_up_sync(0xFFFFFFFF, offset, i);
		if (i <= warpStride) {
			offset += tmp;
		}
	}

	int widInBlock = threadIdx.x / 32;
	__shared__ int warpSums[32];
	if (warpStride == 31) {
		warpSums[widInBlock] = offset;
	}
	__syncthreads();

	if (!widInBlock) {
		int warpOffset = warpSums[warpStride];
		for (int i = 1; i < 32; i *= 2) {
			int tmp = __shfl_up_sync(0xFFFFFFFF, warpOffset, i);
			if (i <= warpStride) {
				warpOffset += tmp;
			}
		}
		warpSums[warpStride] = warpOffset;
	}
	__syncthreads();

	int wo = (widInBlock > 0) ? warpSums[widInBlock - 1] : 0;

	if (tid < N) {
		int tmp = inArr[tid];
		inArr[tid] = offset + wo - tmp;
	}

	if (threadIdx.x == blockDim.x - 1) {
		outArr[blockIdx.x] = warpSums[31];
	}
}

/// <summary>
/// Funkcja kernela dla drugiego algorytmu BFS działającego na GPU. W identyczny sposób wyznacza sumy prefiksowe na najwyższym poziomie hierarchii wyznaczania offsetów zapisu nowo odwiedzonych
/// sąsiadów dla każdego wierzchołka z obecnej kolejki do następnej. Ponieważ jest to najwyższy poziom hierarchii, nie jest konieczne przekazywanie dalej sumy wszystkich składników bloku.
/// </summary>
/// <param name="d_superBlocksCount">Liczba "super bloków"</param>
/// <param name="d_superBlockOffsets">Tablica, do której zapisywane są sumy prefiksowe</param>
/// <returns></returns>
__global__ void computeSuperBlockOffsetsKernel(int d_superBlocksCount, int* d_superBlockOffsets)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int warpStride = threadIdx.x % 32;
	int offset = 0;
	if (tid < d_superBlocksCount) {
		offset = d_superBlockOffsets[tid];
	}

	for (int i = 1; i < 32; i *= 2) {
		int tmp = __shfl_up_sync(0xFFFFFFFF, offset, i);
		if (i <= warpStride) {
			offset += tmp;
		}
	}

	int widInBlock = threadIdx.x / 32;
	__shared__ int warpSums[32];
	if (warpStride == 31) {
		warpSums[widInBlock] = offset;
	}
	__syncthreads();

	if (!widInBlock) {
		int warpOffset = warpSums[warpStride];
		for (int i = 1; i < 32; i *= 2) {
			int tmp = __shfl_up_sync(0xFFFFFFFF, warpOffset, i);
			if (i <= warpStride) {
				warpOffset += tmp;
			}
		}
		warpSums[warpStride] = warpOffset;
	}
	__syncthreads();

	int wo = (widInBlock > 0) ? warpSums[widInBlock - 1] : 0;

	if (tid < d_superBlocksCount) {
		int tmp = d_superBlockOffsets[tid];
		d_superBlockOffsets[tid] = offset + wo - tmp;
	}
}

/// <summary>
/// Funkcja kernela dla drugiego algorytmu BFS działającego na GPU. Na podstawie id wątka określa początek fragmentu tablicy następnej kolejki na podstawie offsetów pochodzących ze wszystkich
/// poziomów hierarchii, by następnie zapisać w niej nowo odwiedzone wierzchołki. Dodatkowo, ostatni wątek nadpisuje rozmiar kolejki dla następnej iteracji.
/// </summary>
/// <param name="d_neighbours">Tablica sąsiadów wszystkich wierzchołków grafu dla reprezentacji CSR</param>
/// <param name="d_offsets">Tablica początków fragmentów tablicy sąsiadów wszystkich wierzchołków grafu dla reprezentacji CSR</param>
/// <param name="d_parents">Tablica służąca do oznaczania wierzchołków, z których bezpośrednio się doszło do tych rozpatrywanych</param>
/// <param name="d_currentFrontier">Kolejka wierzchołków dla danej iteracji</param>
/// <param name="d_nextFrontier">Kolejka zawierające wierzchołki, które będą rozpatrywnae w następnej iteracji</param>
/// <param name="d_currentFrontierSize">Rozmiar obecnej kolejki</param>
/// <param name="d_frontierOffsets">Tablica offsetów wszystkich wierzchołków kolejki w obrębie 1 bloku</param>
/// <param name="d_blockOffsets">Tablica offsetów wszystkich bloków w obrębie 1 "super bloku"</param>
/// <param name="d_superBlockOffsets">Tablica offsetów wszystkich "super bloków"</param>
/// <param name="nextFrontierSize">Rozmiar następnej kolejki</param>
/// <returns></returns>
__global__ void makeNextFrontierKernel(int* d_neighbours, int* d_offsets, int* d_parents, int* d_currentFrontier, int* d_nextFrontier, int d_currentFrontierSize,
	int* d_frontierOffsets, int* d_blockOffsets, int* d_superBlockOffsets, int* nextFrontierSize)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= d_currentFrontierSize) {
		return;
	}
	int btid = tid / 1024;
	int sbtid = btid / 1024;

	int offset = d_frontierOffsets[tid] + d_blockOffsets[btid] + d_superBlockOffsets[sbtid];

	int u = d_currentFrontier[tid];
	int start = d_offsets[u];
	int end = d_offsets[u + 1];
	int count = 0;

	for (int i = start; i < end; i++) {
		int v = d_neighbours[i];
		if (d_parents[v] == u) {
			d_nextFrontier[offset + count] = v;
			count++;
		}
	}

	if (tid == d_currentFrontierSize - 1) {
		*nextFrontierSize = offset + count;
	}
}

/// <summary>
/// Pierwszy algorytm BFS działający na GPU bazujący na wykorzystaniu funkcji atomowych. Synchronizowany jest między kolejnymi poziomami, kończy się, gdy tablica jest pusta lub został przypisany
/// rodzic wierzchołkowi końcowemu, co jest jednoznaczne z odwiedzeniem tego wierzchołka. Po zakończeniu wyznacza najkrótszą ścieżkę między zadanymi wierzchołkami, jeśli istnieje i zapisuje ją
/// do żądanego pliku, zapisując w osobnej linii kolejne wierzchołki, począwszy od startowego aż do końcowego.
/// </summary>
/// <param name="graph">Rozpatrywany graf</param>
/// <param name="start">Wierzchołek startowy</param>
/// <param name="end">Wierzchołek końcowy</param>
/// <param name="outputFilename">Plik, do którego zapisywana jest ścieżka</param>
/// <returns>Znaleziona ścieżka lub pusta lista, jeśli nie istnieje</returns>
std::list<int> BFSGPU1(Graph& graph, int start, int end, const char* outputFilename) {
	std::cout << "GPU1" << std::endl;
	auto timeStart = std::chrono::high_resolution_clock::now();

	d_Ptrs1 dPtrs;
	int frontierSize;
	int* parents = new int[graph.N];
	int* neighbours = new int[graph.EdgeCount];
	int* offsets = new int[graph.N + 1];
	std::copy(graph.neighbours.begin(), graph.neighbours.end(), neighbours);
	std::copy(graph.offsets.begin(), graph.offsets.end(), offsets);
	std::list<int> path;

	for (int i = 0; i < graph.N; i++) {
		parents[i] = -1;
	}
	parents[start] = -2;

	if (cudaCheckStatus1(cudaMalloc((void**)&dPtrs.d_currentFrontier, sizeof(int) * graph.N), dPtrs, "cudaMalloc error", parents, neighbours, offsets)) {
		exit(1);
	}
	if (cudaCheckStatus1(cudaMalloc((void**)&dPtrs.d_nextFrontier, sizeof(int) * graph.N), dPtrs, "cudaMalloc error", parents, neighbours, offsets)) {
		exit(1);
	}
	if (cudaCheckStatus1(cudaMalloc((void**)&dPtrs.d_neighbours, sizeof(int) * graph.EdgeCount), dPtrs, "cudaMalloc error", parents, neighbours, offsets)) {
		exit(1);
	}
	if (cudaCheckStatus1(cudaMalloc((void**)&dPtrs.d_offsets, sizeof(int) * (graph.N + 1)), dPtrs, "cudaMalloc error", parents, neighbours, offsets)) {
		exit(1);
	}
	if (cudaCheckStatus1(cudaMalloc((void**)&dPtrs.d_parents, sizeof(int) * graph.N), dPtrs, "cudaMalloc error", parents, neighbours, offsets)) {
		exit(1);
	}
	if (cudaCheckStatus1(cudaMalloc((void**)&dPtrs.d_nextFrontierSize, sizeof(int)), dPtrs, "cudaMalloc error", parents, neighbours, offsets)) {
		exit(1);
	}

	frontierSize = 1;

	if (cudaCheckStatus1(cudaMemcpy(dPtrs.d_neighbours, neighbours, sizeof(int) * graph.EdgeCount, cudaMemcpyHostToDevice), dPtrs, "cudaMemcpy error", parents, neighbours, offsets)) {
		exit(1);
	}
	if (cudaCheckStatus1(cudaMemcpy(dPtrs.d_offsets, offsets, sizeof(int) * (graph.N + 1), cudaMemcpyHostToDevice), dPtrs, "cudaMemcpy error", parents, neighbours, offsets)) {
		exit(1);
	}
	if (cudaCheckStatus1(cudaMemcpy(dPtrs.d_parents, parents, sizeof(int) * graph.N, cudaMemcpyHostToDevice), dPtrs, "cudaMemcpy error", parents, neighbours, offsets)) {
		exit(1);
	}
	if (cudaCheckStatus1(cudaMemcpy(dPtrs.d_currentFrontier, &start, sizeof(int), cudaMemcpyHostToDevice), dPtrs, "cudaMemcpy error", parents, neighbours, offsets)) { 
		exit(1); 
	}

	auto timeEnd = std::chrono::high_resolution_clock::now();
	auto durationGPU = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart);
	std::cout << std::endl << "Time of memory allocation and copying data: " << durationGPU.count() << " ms." << std::endl;

	timeStart = std::chrono::high_resolution_clock::now();

	dim3 blockDim(1024);

	while (frontierSize) {
		if (cudaCheckStatus1(cudaMemset(dPtrs.d_nextFrontierSize, 0, sizeof(int)), dPtrs, "cudaMemcpy error", parents, neighbours, offsets)) {
			exit(1);
		}

		dim3 vertexGridDim((frontierSize + blockDim.x - 1) * 32 / blockDim.x);
		BFS1Kernel << <vertexGridDim, blockDim >> > (dPtrs.d_neighbours, dPtrs.d_offsets, dPtrs.d_parents, dPtrs.d_currentFrontier,
			dPtrs.d_nextFrontier, frontierSize, dPtrs.d_nextFrontierSize);

		auto cudaStatus = cudaDeviceSynchronize();
		if (cudaCheckStatus1(cudaStatus, dPtrs, "BFS1Kernel error", parents, neighbours, offsets)) {
			std::cerr << cudaStatus;
			exit(1);
		}

		if (cudaCheckStatus1(cudaMemcpy(&frontierSize, dPtrs.d_nextFrontierSize, sizeof(int), cudaMemcpyDeviceToHost), dPtrs, "cudaMemcpy error", parents, neighbours, offsets)) {
			exit(1);
		}
		if (cudaCheckStatus1(cudaMemcpy(parents + end, dPtrs.d_parents + end, sizeof(int), cudaMemcpyDeviceToHost), dPtrs, "cudaMemcpy error", parents, neighbours, offsets)) {
			exit(1);
		}

		if (parents[end] != -1) {
			if (cudaCheckStatus1(cudaMemcpy(parents, dPtrs.d_parents, sizeof(int) * graph.N, cudaMemcpyDeviceToHost), dPtrs, "cudaMemcpy error", parents, neighbours, offsets)) {
				exit(1);
			}
			break;
		}

		std::swap(dPtrs.d_currentFrontier, dPtrs.d_nextFrontier);
	}

	timeEnd = std::chrono::high_resolution_clock::now();
	durationGPU = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart);
	std::cout << std::endl << "Time on GPU (1st algorithm): " << durationGPU.count() << " ms." << std::endl;

	if (parents[end] != -1) {
		int u = end;
		path.push_front(u);
		u = parents[u];
		while (u != -1 && u != -2) {
			;
			path.push_front(u);
			u = parents[u];
		}
	}

	FILE* f = fopen(outputFilename, "w");

	for (int u : path) {
		fprintf(f, "%d\n", u);
	}

	fclose(f);

	freeCudaMemory1(dPtrs);
	delete[] parents;
	delete[] neighbours;
	delete[] offsets;

	return path;
}

/// <summary>
/// Drugi algorytm BFS działający na GPU bazujący na kontrolowanym zapisie nowych wierzchołków do kolejki przy wyznaczaniu dla nich offsetów przy użyciu buforów na różnych poziomach hierarchii. 
/// Synchronizowany jest między kolejnymi poziomami, kończy się, gdy tablica jest pusta lub został przypisany rodzic wierzchołkowi końcowemu, co jest jednoznaczne z odwiedzeniem tego wierzchołka.
/// Po zakończeniu wyznacza najkrótszą ścieżkę między zadanymi wierzchołkami, jeśli istnieje i zapisuje ją do żądanego pliku, zapisując w osobnej linii kolejne wierzchołki, począwszy od startowego 
/// aż do końcowego.
/// </summary>
/// <param name="graph">Rozpatrywany graf</param>
/// <param name="start">Wierzchołek startowy</param>
/// <param name="end">Wierzchołek końcowy</param>
/// <param name="outputFilename">Plik, do którego zapisywana jest ścieżka</param>
/// <returns>Znaleziona ścieżka lub pusta lista, jeśli nie istnieje</returns>
std::list<int> BFSGPU2(Graph& graph, int start, int end, const char* outputFilename)
{
	std::cout << "GPU2" << std::endl;
	auto timeStart = std::chrono::high_resolution_clock::now();

	d_Ptrs2 dPtrs;

	int* parents = new int[graph.N];
	int* neighbours = new int[graph.EdgeCount];
	int* offsets = new int[graph.N + 1];
	std::copy(graph.neighbours.begin(), graph.neighbours.end(), neighbours);
	std::copy(graph.offsets.begin(), graph.offsets.end(), offsets);
	std::list<int> path;

	for (int i = 0; i < graph.N; i++) {
		parents[i] = -1;
	}
	parents[start] = -2;

	if (cudaCheckStatus2(cudaMalloc((void**)&dPtrs.d_currentFrontier, sizeof(int) * graph.N), dPtrs, "cudaMalloc error", parents, neighbours, offsets)) {
		exit(1);
	}
	if (cudaCheckStatus2(cudaMalloc((void**)&dPtrs.d_nextFrontier, sizeof(int) * graph.N), dPtrs, "cudaMalloc error", parents, neighbours, offsets)) {
		exit(1);
	}
	if (cudaCheckStatus2(cudaMalloc((void**)&dPtrs.d_neighbours, sizeof(int) * graph.EdgeCount), dPtrs, "cudaMalloc error", parents, neighbours, offsets)) {
		exit(1);
	}
	if (cudaCheckStatus2(cudaMalloc((void**)&dPtrs.d_offsets, sizeof(int) * (graph.N + 1)), dPtrs, "cudaMalloc error", parents, neighbours, offsets)) {
		exit(1);
	}
	if (cudaCheckStatus2(cudaMalloc((void**)&dPtrs.d_parents, sizeof(int) * graph.N), dPtrs, "cudaMalloc error", parents, neighbours, offsets)) {
		exit(1);
	}
	if (cudaCheckStatus2(cudaMalloc((void**)&dPtrs.d_frontierOffsets, sizeof(int) * (graph.N)), dPtrs, "cudaMalloc error", parents, neighbours, offsets)) {
		exit(1);
	}
	if (cudaCheckStatus2(cudaMalloc((void**)&dPtrs.d_blockOffsets, sizeof(int) * (graph.N / 1024 + 1)), dPtrs, "cudaMalloc error", parents, neighbours, offsets)) {
		exit(1);
	}
	if (cudaCheckStatus2(cudaMalloc((void**)&dPtrs.d_superBlockOffsets, sizeof(int) * 1024), dPtrs, "cudaMalloc error", parents, neighbours, offsets)) {
		exit(1);
	}
	if (cudaCheckStatus2(cudaMalloc((void**)&dPtrs.d_nextFrontierSize, sizeof(int)), dPtrs, "cudaMalloc error", parents, neighbours, offsets)) {
		exit(1);
	}

	int frontierSize = 1;

	if (cudaCheckStatus2(cudaMemcpy(dPtrs.d_neighbours, neighbours, sizeof(int) * graph.EdgeCount, cudaMemcpyHostToDevice), dPtrs, "cudaMemcpy error", parents, neighbours, offsets)) {
		exit(1);
	}
	if (cudaCheckStatus2(cudaMemcpy(dPtrs.d_offsets, offsets, sizeof(int) * (graph.N + 1), cudaMemcpyHostToDevice), dPtrs, "cudaMemcpy error", parents, neighbours, offsets)) {
		exit(1);
	}
	if (cudaCheckStatus2(cudaMemcpy(dPtrs.d_parents, parents, sizeof(int) * graph.N, cudaMemcpyHostToDevice), dPtrs, "cudaMemcpy error", parents, neighbours, offsets)) {
		exit(1);
	}
	if (cudaCheckStatus2(cudaMemcpy(dPtrs.d_currentFrontier, &start, sizeof(int), cudaMemcpyHostToDevice), dPtrs, "cudaMemcpy error", parents, neighbours, offsets)) {
		exit(1);
	}

	auto timeEnd = std::chrono::high_resolution_clock::now();
	auto durationGPU = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart);
	std::cout << std::endl << "Time of memory allocation and copying data: " << durationGPU.count() << " ms." << std::endl;

	timeStart = std::chrono::high_resolution_clock::now();

	dim3 blockDim(1024);

	while (frontierSize) {

		dim3 gridDim1((frontierSize + blockDim.x - 1) / blockDim.x);
		dim3 gridDim2((frontierSize + blockDim.x - 1) * 32 / blockDim.x);
		visitNeighboursKernel << <gridDim2, blockDim >> > (dPtrs.d_neighbours, dPtrs.d_offsets, dPtrs.d_parents, dPtrs.d_currentFrontier, frontierSize, dPtrs.d_frontierOffsets);
		auto cudaStatus = cudaDeviceSynchronize();
		if (cudaCheckStatus2(cudaStatus, dPtrs, "visitNeighboursKernel error", parents, neighbours, offsets)) {
			std::cerr << cudaStatus;
			exit(1);
		}
		computeOffsetsKernel << <gridDim1, blockDim >> > (frontierSize, dPtrs.d_frontierOffsets, dPtrs.d_blockOffsets);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaCheckStatus2(cudaStatus, dPtrs, "computeFrontierOffsetsKernel error", parents, neighbours, offsets)) {
			std::cerr << cudaStatus;
			exit(1);
		}
		int numBlock = (frontierSize + 1023) / 1024;
		dim3 blockGridDim((numBlock + 1023) / blockDim.x);
		computeOffsetsKernel << <blockGridDim, blockDim >> > (numBlock, dPtrs.d_blockOffsets, dPtrs.d_superBlockOffsets);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaCheckStatus2(cudaStatus, dPtrs, "computeBlockOffsetsKernel error", parents, neighbours, offsets)) {
			std::cerr << cudaStatus;
			exit(1);
		}

		int numSuperBlock = (numBlock + 1023) / 1024;
		computeSuperBlockOffsetsKernel << <1, blockDim >> > (numSuperBlock, dPtrs.d_superBlockOffsets);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaCheckStatus2(cudaStatus, dPtrs, "computeSuperBlockOffsetsKernel error", parents, neighbours, offsets)) {
			std::cerr << cudaStatus;
			exit(1);
		}

		makeNextFrontierKernel << <gridDim1, blockDim >> > (dPtrs.d_neighbours, dPtrs.d_offsets, dPtrs.d_parents, dPtrs.d_currentFrontier, dPtrs.d_nextFrontier, frontierSize,
			dPtrs.d_frontierOffsets, dPtrs.d_blockOffsets, dPtrs.d_superBlockOffsets, dPtrs.d_nextFrontierSize);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaCheckStatus2(cudaStatus, dPtrs, "makeNextFrontierKernel error", parents, neighbours, offsets)) {
			std::cerr << cudaStatus;
			exit(1);
		}

		if (cudaCheckStatus2(cudaMemcpy(parents + end, dPtrs.d_parents + end, sizeof(int), cudaMemcpyDeviceToHost), dPtrs, "cudaMemcpy error", parents, neighbours, offsets)) {
			exit(1);
		}

		if (parents[end] != -1) {
			if (cudaCheckStatus2(cudaMemcpy(parents, dPtrs.d_parents, sizeof(int) * graph.N, cudaMemcpyDeviceToHost), dPtrs, "cudaMemcpy error", parents, neighbours, offsets)) {
				exit(1);
			}
			break;
		}

		if (cudaCheckStatus2(cudaMemcpy(&frontierSize, dPtrs.d_nextFrontierSize, sizeof(int), cudaMemcpyDeviceToHost), dPtrs, "cudaMemcpy error", parents, neighbours, offsets)) {
			exit(1);
		}

		std::swap(dPtrs.d_currentFrontier, dPtrs.d_nextFrontier);
	}

	timeEnd = std::chrono::high_resolution_clock::now();
	durationGPU = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart);
	std::cout << std::endl << "Time on GPU (2nd algorithm): " << durationGPU.count() << " ms." << std::endl;

	if (parents[end] != -1) {
		int u = end;
		path.push_front(u);
		u = parents[u];
		while (u != -1 && u != -2) {
			path.push_front(u);
			u = parents[u];
		}
	}

	FILE* f = fopen(outputFilename, "w");

	for (int u : path) {
		fprintf(f, "%d\n", u);
	}

	fclose(f);

	freeCudaMemory2(dPtrs);
	delete[] parents;
	delete[] neighbours;
	delete[] offsets;

	return path;
}

/// <summary>
/// Funkcja pomocnicza do kontroli błędów funkcji CUDA dla BFSGPU1. Przy zgłoszonym błędzie zwalnia zaalokowaną pamięć.
/// </summary>
/// <param name="cudaStatus">Status do rozpatrzenia</param>
/// <param name="dPtrs">Struktura device pointerów</param>
/// <param name="errorMessage">Wiadomość wyświetlana przy zgłoszonym błędzie</param>
/// <param name="d_frontierOffsets"></param>
/// <param name="d_blockOffsets"></param>
/// <param name="d_superBlockOffsets"></param>
int cudaCheckStatus1(cudaError_t cudaStatus, d_Ptrs1& dPtrs, const char* errorMessage, int* parents, int* neighbours, int* offsets) {
	if (cudaStatus != cudaSuccess) {
		std::cout << errorMessage;
		freeCudaMemory1(dPtrs);
		delete[] parents;
		delete[] neighbours;
		delete[] offsets;
		exit(1);
	}
	return 0;
}

/// <summary>
/// Czyści zaalokowaną pamięć na karcie graficznej dla BFSGPU1.
/// </summary>
/// <param name="dPtrs">Struktura device pointerów</param>
/// <param name="d_frontierOffsets"></param>
/// <param name="d_blockOffsets"></param>
/// <param name="d_superBlockOffsets"></param>
void freeCudaMemory1(d_Ptrs1& dPtrs) {
	cudaFree(dPtrs.d_currentFrontier);
	cudaFree(dPtrs.d_neighbours);
	cudaFree(dPtrs.d_nextFrontier);
	cudaFree(dPtrs.d_nextFrontierSize);
	cudaFree(dPtrs.d_offsets);
	cudaFree(dPtrs.d_parents);
}

/// <summary>
/// Funkcja pomocnicza do kontroli błędów funkcji CUDA dla BFSGPU2. Przy zgłoszonym błędzie zwalnia zaalokowaną pamięć.
/// </summary>
/// <param name="cudaStatus">Status do rozpatrzenia</param>
/// <param name="dPtrs">Struktura device pointerów</param>
/// <param name="errorMessage">Wiadomość wyświetlana przy zgłoszonym błędzie</param>
/// <param name="d_frontierOffsets"></param>
/// <param name="d_blockOffsets"></param>
/// <param name="d_superBlockOffsets"></param>
int cudaCheckStatus2(cudaError_t cudaStatus, d_Ptrs2& dPtrs, const char* errorMessage, int* parents, int* neighbours, int* offsets) {
	if (cudaStatus != cudaSuccess) {
		std::cout << errorMessage;
		freeCudaMemory2(dPtrs);
		delete[] parents;
		delete[] neighbours;
		delete[] offsets;
		exit(1);
	}
	return 0;
}

/// <summary>
/// Czyści zaalokowaną pamięć na karcie graficznej dla BFSGPU2.
/// </summary>
/// <param name="dPtrs">Struktura device pointerów</param>
/// <param name="d_frontierOffsets"></param>
/// <param name="d_blockOffsets"></param>
/// <param name="d_superBlockOffsets"></param>
void freeCudaMemory2(d_Ptrs2& dPtrs) {
	cudaFree(dPtrs.d_currentFrontier);
	cudaFree(dPtrs.d_neighbours);
	cudaFree(dPtrs.d_nextFrontier);
	cudaFree(dPtrs.d_nextFrontierSize);
	cudaFree(dPtrs.d_offsets);
	cudaFree(dPtrs.d_parents);
	cudaFree(dPtrs.d_frontierOffsets);
	cudaFree(dPtrs.d_blockOffsets);
	cudaFree(dPtrs.d_superBlockOffsets);
}