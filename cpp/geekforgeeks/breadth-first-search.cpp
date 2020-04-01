#include <assert.h>
#include <iostream>
#include <list>
#include <queue>
#include <vector>

class Graph {
public:
	Graph(int v);
	void addEdge(int v, int w);
	void breadthFirstSearch(int v);

private:
	int vertices;
	std::vector<std::list<int>> adj;
};

Graph::Graph(int v) {
	vertices = v;
	adj.resize(v);
}

void Graph::addEdge(int v, int w) {
	assert(v >= 0 && v < vertices && w >= 0 && w < vertices);
	adj[v].push_back(w);
}

void Graph::breadthFirstSearch(int v) {
	assert(v >= 0 && v < vertices);

	std::vector<bool> visited(vertices, false);
	std::queue<int> q;

	visited[v] = true;
	q.push(v);

	while (!q.empty()) {
		v = q.front();
		std::cout << v << " ";
		q.pop();

		for (auto const & w : adj[v]) {
			if (!visited[w]) {
				visited[w] = true;
				q.push(w);
			}
		}
	}

	std::cout << std::endl;
}

int main() {
	Graph g(4);

	g.addEdge(0, 1);
	g.addEdge(0, 2);
	g.addEdge(1, 2);
	g.addEdge(2, 0);
	g.addEdge(2, 3);
	g.addEdge(3, 3);

	std::cout << "Breadth first search traversal:" << std::endl;
	g.breadthFirstSearch(2);

	return 0;
}
