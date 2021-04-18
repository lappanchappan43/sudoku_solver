def print_sudoku(grid):
    for row in range(9):
        for col in range(9):
            print(grid[row][col], end=' ')
        print()

'''
    Check if there is any empty index on sudoku grid
'''
def empty_index(grid, index: list):
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                index[0] = row
                index[1] = col
                return True
    return False

'''
    Check if value exist in row
'''
def num_in_row(grid, row, val):
    for i in range(9):
        if grid[row][i] == val:
            return True
    return False

'''
    Check if value exist in column
'''
def num_in_col(grid, col, val):
    for i in range(9):
        if grid[i][col] == val:
            return True
    return False

'''
    Check if value exist in 3x3 grid
'''
def num_in_grid(grid, row, col, val):
    for i in range(3):
        for j in range(3):
            if grid[i+row][j+col] == val:
                return True
    return False

def is_safe(grid, row, col, val):
    return not num_in_row(grid, row, val) and \
            not num_in_col(grid, col, val) and \
            not num_in_grid(grid, row-(row%3), col-(col%3), val)

def solver(grid):
    index = [0, 0]
    if not empty_index(grid, index):
        return True
    
    row, col = index[0], index[1]

    for val in range(1, 10):
        if is_safe(grid, row, col, val):
            grid[row][col] = val

            if solver(grid):
                return True

            grid[row][col] = 0
    
    return False


if __name__ == '__main__':
    '''
    grid = [
            [3, 0, 6, 5, 0, 8, 4, 0, 0],
            [5, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 8, 7, 0, 0, 0, 0, 3, 1],
            [0, 0, 3, 0, 1, 0, 0, 8, 0],
            [9, 0, 0, 8, 6, 3, 0, 0, 5],
            [0, 5, 0, 0, 9, 0, 6, 0, 0],
            [1, 3, 0, 0, 0, 0, 2, 5, 0],
            [0, 0, 0, 0, 0, 0, 0, 7, 4],
            [0, 0, 5, 2, 0, 6, 3, 0, 0]
        ]

    grid = [[5, 3, 0, 0, 7, 0, 0, 0, 0], 
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]]

    grid =[
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
    '''

    solver(grid)

    print_sudoku(grid)