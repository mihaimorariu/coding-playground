#include <assert.h>
#include <iostream>
#include <list>
#include <stack>
#include <vector>

class Graph {
public:
	Graph(int v);
	void addEdge(int v, int w);
	void depthFirstSearch(int v);

private:
	int vertices;
	std::vector<std::list<int>> adj;

	void depthFirstSearch(int v, std::vector<bool> & visited);
};

Graph::Graph(int v) {
	vertices = v;
	adj.resize(v);
}

void Graph::addEdge(int v, int w) {
	assert(v >= 0 && v < vertices && w >= 0 && w < vertices);
	adj[v].push_back(w);
}

void Graph::depthFirstSearch(int v, std::vector<bool> & visited) {
	visited[v] = true;
	std::cout << v << " ";

	for (auto const & w : adj[v]) {
		if (!visited[w]) {
			depthFirstSearch(w, visited);
		}
	}
}

void Graph::depthFirstSearch(int v) {
	std::vector<bool> visited(vertices, false);
	depthFirstSearch(v, visited);
	std::cout << std::endl;
}

//void Graph::depthFirstSearch(int v) {
	//assert(v >= 0 && v < vertices);

	//std::vector<bool> visited(vertices, false);
	//std::stack<int> s;

	//visited[v] = true;
	//s.push(v);

	//while (!s.empty()) {
		//v = s.top();

		//std::cout << v << " ";
		//s.pop();

		//for (auto const & w : adj[v]) {
			//if (!visited[w]) {
				//visited[w] = true;
				//s.push(w);
			//}
		//}
	//}

	//std::cout << std::endl;
//}

int main() {
	Graph g(4);

	g.addEdge(0, 1);
	g.addEdge(0, 2);
	g.addEdge(1, 2);
	g.addEdge(2, 0);
	g.addEdge(2, 3);
	g.addEdge(3, 3);

	std::cout << "Depth first search traversal:" << std::endl;
	g.depthFirstSearch(2);

	return 0;
}
