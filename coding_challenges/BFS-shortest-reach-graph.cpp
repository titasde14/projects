/*
Created by Titas De

Consider an undirected graph consisting of  nodes where each node is labeled from  to  and the edge between any two nodes is always of length . We define node  to be the starting position for a BFS.

Given  queries in the form of a graph and some starting node, , perform each query by calculating the shortest distance from starting node  to all the other nodes in the graph. Then print a single line of  space-separated integers listing node 's shortest distance to each of the  other nodes (ordered sequentially by node number); if  is disconnected from a node, print  as the distance to that node.

Input Format

The first line contains an integer, , denoting the number of queries. The subsequent lines describe each query in the following format:

The first line contains two space-separated integers describing the respective values of  (the number of nodes) and  (the number of edges) in the graph.
Each line  of the  subsequent lines contains two space-separated integers,  and , describing an edge connecting node  to node .
The last line contains a single integer, , denoting the index of the starting node.
Constraints

Output Format

For each of the  queries, print a single line of  space-separated integers denoting the shortest distances to each of the  other nodes from starting position . These distances should be listed sequentially by node number (i.e., ), but should not include node . If some node is unreachable from , print  as the distance to that node.

Sample Input

2
4 2
1 2
1 3
1
3 1
2 3
2
Sample Output

6 6 -1
-1 6

*/


#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <queue>
#include <unordered_set>
using namespace std;

class Graph
{
public:
    vector<int> *adj;
    int V;
    Graph(int V)
    {
        this->V = V;
        adj = new vector<int>[V];
    }

    void add_edge(int u, int v)
    {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> shortest_reach(int start, int n)
    {
        vector<int> distances(n,-1);
        queue<int> que;
        unordered_set<int> seen;

        que.push(start);
        distances[start] = 0;

        seen.insert(start);
        while(!que.empty())
        {

            int cur = que.front();
            que.pop();
            for(int node : adj[cur])
            {
                if (seen.find(node) == seen.end() )
                {
                    que.push(node);
                    seen.insert(node);
                    distances[node] = distances[cur] + 6;
                }
            }
        }
        return distances;
    }

};

int main()
{
    int queries;
    cin >> queries;

    for (int t = 0; t < queries; t++)
    {

        int n, m;
        cin >> n;
        // Create a graph of size n where each edge weight is 6:
        Graph graph(n);
        cin >> m;
        // read and set edges
        for (int i = 0; i < m; i++)
        {
            int u, v;
            cin >> u >> v;
            u--, v--;
            // add each edge to the graph
            graph.add_edge(u, v);
        }
        int startId;
        cin >> startId;
        startId--;
        // Find shortest reach from node s
        vector<int> distances = graph.shortest_reach(startId, n);

        for (int i = 0; i < distances.size(); i++)
        {
            if (i != startId)
            {
                cout << distances[i] << " ";
            }
        }
        cout << endl;
    }

    return 0;
}