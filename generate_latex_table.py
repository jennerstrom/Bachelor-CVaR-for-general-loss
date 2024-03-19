import numpy as np

#miw = [
# [[2.68112461e+05, 0.00000000e+00,1.38900280e-02],
#  [4.76902700e+05, 0.00000000e+00,4.29105759e-03],
#  [3.09683822e+05, 0.00000000e+00,3.07512283e-03],
#  [4.38848557e+05, 0.00000000e+00,3.76892090e-03],
#  [2.59017388e+05, 0.00000000e+00,1.39319897e-02],
#  [4.67787216e+05, 0.00000000e+00,7.47299194e-03],
#  [3.52246497e+05, 0.00000000e+00,4.57000732e-03],
#  [5.50652033e+05, 0.00000000e+00,4.81700897e-03],
#  [3.47799571e+05, 0.00000000e+00,9.54794884e-03],
#  [5.48300207e+05, 0.00000000e+00,5.95498085e-03],
#  [3.18156164e+05, 0.00000000e+00,9.15789604e-03],
#  [4.64647006e+05, 0.00000000e+00,1.34959221e-02]],
#
# [[2.28400683e+05, 0.00000000e+00,6.62302971e-03],
#  [3.21298674e+05, 0.00000000e+00,3.79610062e-03],
#  [2.56845012e+05, 0.00000000e+00,3.53598595e-03],
#  [3.89814781e+05, 0.00000000e+00,9.10902023e-03],
#  [2.35021676e+05, 0.00000000e+00,1.09391212e-02],
#  [3.80520011e+05, 0.00000000e+00,2.15342045e-02],
#  [3.09904182e+05, 0.00000000e+00,4.29916382e-03],
#  [4.13980924e+05, 0.00000000e+00,7.11083412e-03],
#  [2.86437753e+05,0.00000000e+00,9.11092758e-03],
#  [4.18171463e+05,0.00000000e+00,6.26802444e-03],
#  [2.84548715e+05,0.00000000e+00,1.85790062e-02],
#  [4.56173654e+05,0.00000000e+00,1.44031048e-02]]
#]

#origins = [5,10]
#destinations = [15,20]
#scenarios = [8,12,25]
#commodities = [1,2]
#alpha = 0.95

#Den fungerer kun for to niveauer af L - tror at 3 ville gøre tabellen for bred til A4

def generate_table(origins,destinations,scenarios,commodities,alpha,miw):
    f = open("table.txt", "w")

    my_list = []
    for i in range(len(origins)):
        for j in range(len(destinations)):
            for s in range(len(scenarios)):
                my_list.append([origins[i],destinations[j],scenarios[s]])

    for index,tup in enumerate(my_list):
        if index % 6 != 0:
            my_list[index][0] = ''
        if index % 3 != 0:
            my_list[index][1] = ''

    cols = len(miw[0][0])*len(miw)
    col_string = 'l'*(3+cols)

    f.write(r'\begin{table}[]' + '\n')
    f.write(r'\begin{tabular}{' + col_string + '}' + '\n')
    f.write(r'\hline' + '\n')
    f.write(r'$|I|$ & $|J|$ & $|S|$')

    for i in range(1,len(miw)+1):
        f.write(f' & $L={i}$')
        for j in range(len(miw[0][0])-1):
            f.write(f' & ')
    f.write(r'\\ \hline' + '\n')
    f.write('& & ')
    for i in range(len(miw)):
        f.write(f' & Obj. & Gap & Runtime')
    f.write(r'\\ \hline' + '\n')

    for index,(commod1,commod2) in enumerate(zip(miw[0],miw[1])):
        f.write(f'{my_list[index][0]} & {my_list[index][1]} & {my_list[index][2]} & ')
        for i in range(len(commod1)):
            f.write(f'{commod1[i]} & ')
        for i in range(len(commod2)):
            if i == len(commod2)-1:
                f.write(f'{commod2[i]} \n')
            else:
                f.write(f'{commod2[i]} & ')
        f.write(r'\\' + '\n')
    print(my_list)
    f.write(r'\end{tabular}' + '\n')
    f.write(r'\end{table}')
    f.close()