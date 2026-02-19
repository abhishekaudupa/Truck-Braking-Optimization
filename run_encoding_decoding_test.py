import math

def encode_network(w_ih,w_ho,w_max):

    chromosome = []
    weight_matrices = [w_ih, w_ho]

    for weight_matrix in weight_matrices:
        row_count = len(weight_matrix)
        col_count = len(weight_matrix[0])
        for i in range(row_count):
            for j in range(col_count):
                w = weight_matrix[i][j]
                gene = (w + w_max) / (2 * w_max)
                chromosome.append(gene)

    return chromosome

def decode_chromosome(chromosome, n_i, n_h, n_o, w_max):
    w_ih = []
    w_ho = []

    gene_counter = 0
    
    #decode w_ih
    for i in range(n_h):
        row = []
        for j in range(n_i + 1):
            gene = chromosome[gene_counter]
            w = w_max * (2 * gene - 1)
            row.append(w)
            gene_counter += 1
        w_ih.append(row)

    # decode w_ho
    for i in range(n_o):
        row = []
        for j in range(n_h + 1):
            gene = chromosome[gene_counter]
            w = w_max * (2 * gene - 1)
            row.append(w)
            gene_counter += 1
        w_ho.append(row)

    return w_ih, w_ho

############################################################
# Main program:
############################################################

# The maximum (absolute) value of weights and biases. Thus, they take values in
# the range [-w_max,w_max]
w_max = 5;

# Sample network of size 3-3-2. Note the the number of rows in w_ih MUST be
# equal to the number of columns in w_ho, minus 1; see also the definition of nH below.
#
# Note: Your encoding and decoding methods should work for any values of nI, nH, and nO,
# not just for the example below! Thus, test your encoding and decoding functions by
# defining different set of matrices w_ih and w_ho (fulfilling the criterion on nH, see below)
#
w_ih = [ [2, 1, -3, 1], [5, -2, 1, 4], [3, 0, 1, 2]];
w_ho = [[1, 0, -4, 3], [4, -2, 0, 1]];
n_i = len(w_ih[0])-1
n_h = len(w_ih) # % must be equal to len(w_ho[0])-1, for a valid set of matrices for an FFNN
n_o = len(w_ho)

chromosome = encode_network(w_ih,w_ho,w_max)
[new_w_ih, new_w_ho] = decode_chromosome(chromosome,n_i,n_h,n_o,w_max)

error_count = 0
tolerance = 0.00000001
for i in range(n_h):
  for j in range(n_i+1):
    difference = abs(w_ih[i][j]-new_w_ih[i][j])
    if (difference > tolerance):
      print("Error for element " + str(i) + " , " + str(j) + " in wIH")
      error_count += 1

for i in range(n_o):
  for j in range(n_h+1):
    difference = abs(w_ho[i][j]-new_w_ho[i][j])
    if (difference > tolerance):
      print("Error for element " + str(i) + " , " + str(j) + " in wHO")
      error_count += 1

if (error_count == 0):
  print("Test OK")
else:
  print("Test failed")
input(f'Press return to exit')
