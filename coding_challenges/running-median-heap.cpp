/*
Created by Titas De

The median of a dataset of integers is the midpoint value of the dataset for which an equal number of integers are less than and greater than the value. To find the median, you must first sort your dataset of integers in non-decreasing order, then:

If your dataset contains an odd number of elements, the median is the middle element of the sorted sample. In the sorted dataset ,  is the median.
If your dataset contains an even number of elements, the median is the average of the two middle elements of the sorted sample. In the sorted dataset ,  is the median.
Given an input stream of  integers, you must perform the following task for each  integer:

Add the  integer to a running list of integers.
Find the median of the updated list (i.e., for the first element through the  element).
Print the list's updated median on a new line. The printed value must be a double-precision number scaled to decimal place (i.e.,  format).
Input Format

The first line contains a single integer, , denoting the number of integers in the data stream. 
Each line  of the  subsequent lines contains an integer, , to be added to your list.

Constraints

Output Format

After each new integer is added to the list, print the list's updated median on a new line as a single double-precision number scaled to  decimal place (i.e.,  format).

Sample Input

6
12
4
5
3
8
7
Sample Output

12.0
8.0
5.0
4.5
5.0
6.0

*/

#include <map>
#include <set>
#include <list>
#include <cmath>
#include <ctime>
#include <deque>
#include <queue>
#include <stack>
#include <string>
#include <bitset>
#include <cstdio>
#include <limits>
#include <vector>
#include <climits>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <unordered_map>

using namespace std;

#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <vector>
#include <math.h>

using namespace std;

bool compare_less(const int& a, const int& b)
{
    return b < a;
}

bool compare_greater(const int& a, const int& b)
{
    return a < b;
}

int main(){
    int x, n;
    cin >> n;
    vector<int> v0, v1;
    v0.reserve(n);
    v1.reserve(n);

    float m = nanf("");

    for(int i = 0; i < n; i++) {
        cin >> x;

        if (isnan(m) || (float)x <= m)
        {
            v0.push_back(x);
            push_heap(v0.begin(), v0.end(), compare_greater);
        }
        else
        {
            v1.push_back(x);
            push_heap(v1.begin(), v1.end(), compare_less);
        }

        while (v1.size() > v0.size()+1)
        {
            x = v1[0];
            pop_heap(v1.begin(), v1.end(), compare_less);
            v1.pop_back();
            v0.push_back(x);
            push_heap(v0.begin(), v0.end(), compare_greater);
        }
        while (v0.size() > v1.size()+1)
        {
            x = v0[0];
            pop_heap(v0.begin(), v0.end(), compare_greater);
            v0.pop_back();
            v1.push_back(x);
            push_heap(v1.begin(), v1.end(), compare_less);
        }
        if (v0.size() > v1.size())
            m = v0[0];
        else if (v1.size() > v0.size())
            m = v1[0];
        else
            m = (v0[0]+v1[0])/2.0f;
        
        printf("%.1f\n", m);
    }

    return 0;
}




