def print_sudoku(grid):
    for row in range(9):
        for col in range(9):
            print(grid[row][col], end=' ')
        print()

'''
    Check if there is any empty index on sudoku grid
'''
def empty_index(grid, index: list, matrix: list):
    for row in range(matrix[0]):
        for col in range(matrix[1]):
            if grid[row][col] == 0:
                index[0] = row
                index[1] = col
                return True
    return False

'''
    Check if value exist in row
'''
def num_in_row(grid, row, val, row_num):
    for i in range(row_num):
        if grid[row][i] == val:
            return True
    return False

'''
    Check if value exist in column
'''
def num_in_col(grid, col, val, col_num):
    for i in range(col_num):
        # print(grid[i][col], i)
        if grid[i][col] == val:
            return True
    return False

'''
    Check if value exist in 3x3 grid
'''
def num_in_grid(grid, row, col, val):
    for r in range(3):
        for c in range(3):
            if grid[r+row][c+col] == val:
                return True
    return False

def is_safe(grid, row, col, val, matrix:list):
    row_num, col_num = matrix[0], matrix[1]
    return not num_in_row(grid, row, val, row_num=row_num) and \
            not num_in_col(grid, col, val, col_num=col_num) and \
            not num_in_grid(grid, row-(row%3), col-(col%3), val)

def solver(grid, grid_matrix=[9, 9]):
    index = [0, 0]
    if not empty_index(grid, index, matrix=grid_matrix):
        return True
    
    row, col = index[0], index[1]

    for val in range(1, grid_matrix[0]+1):
        if is_safe(grid, row, col, val, matrix=grid_matrix):
            grid[row][col] = val

            if solver(grid):
                return True

            grid[row][col] = 0
    
    return False


if __name__ == '__main__':
    grid = [
            [7, 8, 0, 4, 0, 0, 1, 2, 0],
            [6, 0, 0, 0, 7, 5, 0, 0, 9],
            [0, 0, 0, 6, 0, 1, 0, 7, 8],
            [0, 0, 7, 0, 4, 0, 2, 6, 0],
            [0, 0, 1, 0, 5, 0, 9, 3, 0],
            [9, 0, 4, 0, 6, 0, 0, 0, 5],
            [0, 7, 0, 3, 0, 0, 0, 1, 2],
            [1, 2, 0, 0, 0, 7, 4, 0, 0],
            [0, 4, 9, 2, 0, 6, 0, 0, 7]
        ]
    solver(grid)

    print_sudoku(grid)