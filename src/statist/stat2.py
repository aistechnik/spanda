
def get_stat2(fname):
    print('File: ' + fname)
    finp = open('input_data.csv', 'r')
    fout = open('stat2_data.csv', 'w')
    count = 0
    couno = 0    
 
    title = finp.readline()
    fout.write('time,durat,value\n')
    
    tima = -1.0
    time = 0.0
    valt = -1.0
 

    for line in finp:
        pole = line.split(',')
        
        if float(pole[1]) != valt:
            if tima == -1.0:
                tima = float(pole[0])
                valt = float(pole[1])
            else:
                durat = float(pole[0]) - tima
                if durat > 1.4:
                    fout.write(str(couno) + ',' + str(round(durat,4)) + ',' + str(round(valt,4)) + '\n')
                    couno += 1 

                tima = float(pole[0])
                valt = float(pole[1])
                           
         
        count += 1
       
     
    print(count, couno)
    #finp.close()
    fout.close()
