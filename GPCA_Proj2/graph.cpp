#include "graph.h"
#include <fstream>
#include <queue>
#include <cmath>

void Graph::pushEdge(int start, int end) {
	int max = std::max(start, end);

	if (max >= this->graph.size()) {
		this->graph.resize(max + 1);
	}
	this->graph[start].push_back(end);
}

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

bool Graph::readGraphFromFile(const char* filename) {
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
			pushEdge(start, end);
			this->EdgeCount++;
		}
	}
	transformGraph();
	return true;
}

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
