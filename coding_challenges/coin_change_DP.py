
'''
Created by Titas De

Given a number of dollars, , and a list of dollar values for  distinct coins, , find and print the number of different ways you can make change for  dollars if each coin is available in an infinite quantity.

Hints:

You can solve this problem recursively, but you must optimize your solution to eliminate overlapping subproblems using Dynamic Programming if you wish to pass all test cases. More specifically, think of ways to store the checked solutions and use the stored values to avoid repeatedly calculating the same values.
Think about the degenerate cases: 
How many ways can you make change for  dollars?
How many ways can you make change for less than  dollars if you have no coins?
If you are having trouble defining the storage for your precomputed values, then think about it in terms of the base case .
Input Format

The first line contain two space-separated integers describing the respective values of  and . 
The second line contains  space-separated integers describing the respective values of , where each integer denotes the dollar value of a distinct coin available in an infinite quantity.


'''

import sys

def make_change(coins, n):
    results = [0 for _ in range(n + 1)]
    results[0] = 1
    for coin in coins:
        for i in range(coin, n + 1):
            results[i] += results[i - coin]
    return results[n]

n,m = raw_input().strip().split(' ')
n,m = [int(n),int(m)]
coins = map(int,raw_input().strip().split(' '))
print make_change(coins, n)