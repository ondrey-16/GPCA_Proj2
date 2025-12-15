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

bool Graph::readGraphFromFile(std::string filename) {
	std::ifstream file(filename);

	if (!file.is_open()) {
		std::cerr << "Cannot open the file\n";
		return false;
	}

	int start, end;
	while (file >> start >> end) {
		pushEdge(start, end);
	}

	if (!file.eof()) {
		std::cerr << "Reading from file error\n";
		return false;
	}

	return true;
}

std::list<int> Graph::BFSCPU(int start, int end) {
	if (start == end) {
		return std::list<int>{start};
	}

	std::vector<bool> visited(this->graph.size(), false);
	std::vector<int> parents(this->graph.size(), -1);

	std::list<int> path;
	std::queue<int> Q;
	Q.push(start);
	visited[start] = true;

	while (!Q.empty()) {
		int v = Q.front();
		Q.pop();
		if (v == end) {
			break;
		}

		for (int w : this->graph[v]) {
			if (!visited[w]) {
				Q.push(w);
				visited[w] = true;
				parents[w] = v;
			}
		}
	}

	if (!visited[end]) {
		return path;
	}

	int u = end;

	while (u != -1) {
		path.push_front(u);
		u = parents[u];
	}

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
