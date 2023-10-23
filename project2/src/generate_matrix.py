import random

def gen_matrix(row_num, col_num, name):
    # row_num = 100
    # col_num = 100
    maximum_limit = 100.0

    with open("matrix_{}.txt".format(name), "w") as matrix_file:
        matrix_file.write(f"{row_num} {col_num}\n")
        for i in range(row_num):
            for j in range(col_num):
                random_number = int(random.uniform(0.0, maximum_limit))
                if j < col_num - 1:
                    matrix_file.write(f"{random_number} ")
                else:
                    matrix_file.write(f"{random_number}\n")
                    
def gen_matrix_pairs(m, k, n):
    gen_matrix(m, k, "test0")
    gen_matrix(k, n, "test1")
    
    
# def compare():
    
    
    
            
