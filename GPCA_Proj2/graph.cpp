#define _CRT_SECURE_NO_WARNINGS

#include "graph.h"
#include <fstream>
#include <queue>
#include <cmath>

/// <summary>
/// Modyfikuje listê s¹siedztwa grafu o now¹ dodan¹ krawêdŸ. Je¿eli trzeba, rozszerza odpowiednio listê do rozmiaru odpowiadaj¹cego maksymalnemu numerowi wierzcho³ka w grafie.
/// Je¿eli format czytanego grafu jest dla grafu nieskierowanego, dodaje jeszcze krawêdŸ przeciwn¹.
/// </summary>
/// <param name="start">Wierzcho³ek pocz¹tkowy</param>
/// <param name="end"Wierzcho³ek koñcowy></param>
/// <param name="format">Format czytanego grafu</param>
void Graph::pushEdge(int start, int end, const char* format) {
	int max = std::max(start, end);

	if (max >= this->graph.size()) {
		this->graph.resize(max + 1);
	}
	this->graph[start].push_back(end);
	if (strcmp(format, "txt_undir") == 0) {
		this->graph[end].push_back(start);
	}
}

/// <summary>
/// Dokonuje przekszta³cenia reprezentacji grafu z list s¹siedztwa do reprezentacji CSR, inicjalizuj¹c odpowiednio tablice offsets i neighbours.
/// </summary>
void Graph::transformGraph() {
	this->N = (int)this->graph.size();
	this->offsets.resize(this->N + 1);
	int count = 0;
	for (int i = 0; i < this->N; i++) {
		offsets[i] = count;
		for (int j : this->graph[i]) {
			neighbours.push_back(j);
			count++;
		}
	}
	offsets[this->N] = count;
}


/// <summary>
/// Dokonuje odczytu pliku z krawêdziami rozpatrywanego grafu, inicjalizuj¹c przy tym listê s¹siedztwa.
/// </summary>
/// <param name="filename">Œcie¿ka do czytanego pliku</param>
/// <param name="format">Format czytanego grafu</param>
/// <returns>True, gdy poprawnie odczytano i zainicjalizowano graf</returns>
bool Graph::readGraphFromFile(const char* filename, const char* format) {
	FILE* file = fopen(filename, "r");

	if (file == nullptr) {
		std::cerr << "Cannot open the file\n";
		return false;
	}

	int start, end;
	char buff[256];
	this->EdgeCount = 0;
	while (fgets(buff, sizeof(buff), file)) {
		if (buff[0] == '#') {
			continue;
		}
		else if (sscanf(buff, "%d %d", &start, &end) == 2) {
			pushEdge(start, end, format);
			this->EdgeCount++;
			if (strcmp(format, "txt_undir") == 0) {
				this->EdgeCount++;
			}
		}
	}
	transformGraph();
	fclose(file);
	return true;
}

/// <summary>
/// Algorytm BFS dzia³aj¹cy na CPU na reprezentacji CSR grafu. Zwraca najkrótsz¹ œcie¿kê pomiêdzy zadanymi wierzcho³kami.
/// </summary>
/// <param name="start">Wierzcho³ek pocz¹tkowy</param>
/// <param name="end"Wierzcho³ek koñcowy></param>
/// <returns>Znaleziona najkrótsza œcie¿ka</returns>
std::list<int> Graph::BFSCPU(int start, int end) {
	if (start == end) {
		return std::list<int>{start};
	}

	int* parents = new int[this->N];

	for (int i = 0; i < this->N; i++) {
		parents[i] = -1;
	}

	std::list<int> path;
	std::queue<int> Q;
	Q.push(start);
	parents[start] = -2;

	while (!Q.empty()) {
		int v = Q.front();
		Q.pop();
		if (v == end) {
			break;
		}

		for (int i = this->offsets[v]; i < this->offsets[v + 1]; i++) {
			int w = this->neighbours[i];
			if (parents[w] == -1) {
				Q.push(w);
				parents[w] = v;
			}
		}
	}

	if (parents[end] == -1) {
		delete[] parents;
		return path;
	}

	int u = end;
	path.push_front(u);
	u = parents[u];
	while (u != -1 && u != -2) {
		path.push_front(u);
		u = parents[u];
	}

	delete[] parents;
	return path;
}

std::ostream& operator<<(std::ostream& out, const Graph& g) {
	for (int v = 0; v < g.graph.size(); v++) {
		out << v << ": [";

		for (int x = 0; x < g.graph[v].size(); x++) {
			out << g.graph[v][x] << " ";
		}
		out << "]\n";
	}

	return out;
}
