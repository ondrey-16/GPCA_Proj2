#include "graph.h"
#include <fstream>

void Graph::pushEdge(int start, int end) {
	if (start >= this->graph.size()) {
		this->graph.resize(start + 1);
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

std::ostream& operator<<(std::ostream& out, const Graph& g)
{
	for (int v = 0; v < g.graph.size(); v++) {
		out << v << ": [";

		for (int x = 0; x < g.graph[v].size(); x++) {
			out << g.graph[v][x] << " ";
		}
		out << "]\n";
	}

	return out;
}
