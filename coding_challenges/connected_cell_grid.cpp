/*
Created by Titas De

Consider a matrix with  rows and  columns, where each cell contains either a  or a  and any cell containing a  is called a filled cell. Two cells are said to be connected if they are adjacent to each other horizontally, vertically, or diagonally; in other words, cell  is connected to cells , , , , , , , and , provided that the location exists in the matrix for that .

If one or more filled cells are also connected, they form a region. Note that each cell in a region is connected to at least one other cell in the region but is not necessarily directly connected to all the other cells in the region.

Task 
Given an  matrix, find and print the number of cells in the largest region in the matrix. Note that there may be more than one region in the matrix.

Input Format

The first line contains an integer, , denoting the number of rows in the matrix. 
The second line contains an integer, , denoting the number of columns in the matrix. 
Each line  of the  subsequent lines contains  space-separated integers describing the respective values filling each row in the matrix.

Constraints

Output Format

Print the number of cells in the largest region in the given matrix.

Sample Input

4
4
1 1 0 0
0 1 1 0
0 0 1 0
1 0 0 0
Sample Output

5
Explanation

The diagram below depicts two regions of the matrix; for each region, the component cells forming the region are marked with an X:

X X 0 0     1 1 0 0
0 X X 0     0 1 1 0
0 0 X 0     0 0 1 0
1 0 0 0     X 0 0 0
The first region has five cells and the second region has one cell. Because we want to print the number of cells in the largest region of the matrix, we print .

*/

#include <iostream>
#include <climits>
#include <vector>
#include <algorithm>
using namespace std;

int getRegion(vector<vector<int>> &a, int i, int j, int n, int m) {

    if (i >= n || i<0 || j>=m || j < 0) return 0;
    if (a[i][j] == 0) return 0;

    int count = a[i][j]--;
    count += getRegion(a, i + 1, j + 1, n, m);
    count += getRegion(a, i - 1, j - 1, n, m);
    count += getRegion(a, i - 1, j, n, m);
    count += getRegion(a, i + 1, j, n, m);
    count += getRegion(a, i, j - 1, n, m);
    count += getRegion(a, i, j + 1, n, m);
    count += getRegion(a, i - 1, j + 1, n, m);
    count += getRegion(a, i + 1, j - 1, n, m);

    return count;
}

int dfs(vector<vector<int>> a, int n, int m) {
    int res = INT_MIN;

    for (int i = 0; i < n;i++) {
        for (int j = 0; j < m;j++) {
            if (a[i][j] == 1) {
                res = max(res, getRegion(a, i, j, n, m));
            }
        }
    }
    return res;
}


int main()
{
    int n; cin >> n;
    int m; cin >> m;
    vector<vector<int>>v(n, vector<int>(m));

    for (int i = 0; i < n;i++) {
        for (int j = 0; j < m;j++) {
            int num; cin >> num;
            v[i][j] = num;
        }
    }
    cout << dfs(v, n, m)<<endl;

    system("pause");
    return 0;
}