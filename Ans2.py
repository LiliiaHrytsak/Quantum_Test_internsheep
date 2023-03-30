from typing import List


def number_islands(m: int, n: int, map_matrix: List[List[int]]) -> int:
    def dfs(matrix_map, i, j):
        if i < 0 or j < 0 or i >= len(matrix_map) or j >= len(matrix_map[0]) or matrix_map[i][j] != 1:
            return
        matrix_map[i][j] = '#'
        dfs(matrix_map, i + 1, j)
        dfs(matrix_map, i - 1, j)
        dfs(matrix_map, i, j + 1)
        dfs(matrix_map, i, j - 1)

    count = 0
    if not map_matrix:
        return 0
    for i in range(m):
        for j in range(n):
            if map_matrix[i][j] == 1:
                dfs(map_matrix, i, j)
                count += 1

    return count


print(number_islands(3, 3, [[0, 1, 0], [0, 0, 0], [0, 1, 1]]))