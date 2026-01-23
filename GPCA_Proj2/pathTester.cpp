#include "pathTester.h"

bool pathTester(Graph& g, std::list<int>& path)
{
	if (path.empty() || path.size() == 1) {
		return true;
	}

	auto end = path.end();
	end--;
	for (auto it = path.begin(); it != end; it++) {
		auto itNext = it;
		itNext++;
		int u = *it, v = *itNext;
		bool vExists = false;
		for (int i = g.offsets[u]; i < g.offsets[u + 1]; i++) {
			if (g.neighbours[i] == v) {
				vExists = true;
				break;
			}
		}
		if (!vExists) {
			return false;
		}
	}

	return true;
}
