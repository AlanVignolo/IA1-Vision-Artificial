import matplotlib.pyplot as plt
from Grafico import Grafico
import copy
from math import dist

class Aestrella():
    def __init__(self,filas,columnas):

        self.Cantfilas = filas
        self.Cantcolumnas = columnas
        self.Obstaculos= self.CalcObstaculos()

    def calcularcamino(self,inicio,final):

        nodoinicial = self.Obstaculos[inicio-1] # Para que sea a partir de 1
        nodofinal = self.Obstaculos[final-1]

        nodofinalp = [nodofinal[0]-1, nodofinal[1]] #Subo 1 cuadradito
        nodoinicial = [nodoinicial[0]-1, nodoinicial[1]]

        nodosabiertos = []
        nodosvisitados = []
        nodoactual = nodoinicial
        nodosvisitados.append(((nodoinicial[0]),(nodoinicial[1]),0))

        if nodoactual == nodofinalp:
            return 0, [nodoinicial, nodofinalp], nodofinalp
        r=0
        while True:

            if nodoactual == nodofinalp:
                break
            r+=1
            iactual = nodoactual[0]
            jactual = nodoactual[1]
            for i in [-1,0, 1]:
                for j in [-1, 0, 1]:
                    inuevo = iactual + i
                    jnuevo = jactual + j
                    if ((inuevo, jnuevo) not in [(sublst[0], sublst[1]) for sublst in nodosabiertos]) and ((inuevo, jnuevo) not in self.Obstaculos) and ((inuevo, jnuevo) not in [(sublst[0], sublst[1]) for sublst in nodosvisitados]) and (inuevo >= 0) and (jnuevo >= 0) and (inuevo <= 11) and (jnuevo <= 11) and abs(i) != abs(j):
                        f = self.heuristica(nodofinalp[0], nodofinalp[1], inuevo, jnuevo) + r
                        nodosabiertos.append((inuevo, jnuevo,f , r, (iactual, jactual)))

            nodosabiertos = sorted(nodosabiertos, key=lambda x: x[2]) #Ordeno por f de menor a mayor
            nodoactual[0],nodoactual[1], *_ = nodosabiertos[0] #Me quedo con el nodo con menor f
            nodosvisitados.append(((nodosabiertos[0][0]),(nodosabiertos[0][1]),(nodosabiertos[0][3]),(nodosabiertos[0][4]))) #Agrego el nodo con menor f a la lista de nodos visitados y le agrego el padre (nodoactual)
            nodosabiertos.pop(0) #Elimino el nodo con menor f de la lista de nodos abiertos
        
        listafinal=[]
        while True:

            for i in range(len(nodosvisitados)-1):

                if len(nodosvisitados)==2:
                    listafinal.insert(0,(nodosvisitados[-1][0],nodosvisitados[-1][1]))
                    listafinal.insert(0,(nodosvisitados[0][0],nodosvisitados[0][1]))
                    return len(listafinal),listafinal,nodofinalp
                
                while nodosvisitados[-i-1][3] != (nodosvisitados[-i-2][0],nodosvisitados[-i-2][1]):
                    nodosvisitados.pop(-i-2)
                listafinal.insert(0,(nodosvisitados[-i-1][0],nodosvisitados[-i-1][1]))

                if i ==len(nodosvisitados)-2: #Si llego al final de la lista de nodos visitados
                    listafinal.insert(0,(nodosvisitados[0][0],nodosvisitados[0][1]))
                    return len(listafinal),listafinal,nodofinalp
        
    def dibujar(self,nodosvisitados,nodofinalp):    
        Grafico1=Grafico(self.Cantfilas*6,self.Cantcolumnas*4,self.Obstaculos)
        Grafico1.dibujar_grafico(nodosvisitados,nodofinalp)

    def GraficarCamino (self,lista):

        nodosvisitados = []
        finales = []
        for i in range(len(lista)-1):
            nodosvisitados.append((self.calcularcamino(lista[i],lista[i+1]))[1])
        Grafico2=Grafico(12,12,self.Obstaculos)
        for x in lista:
            finales.append(self.Obstaculos[x-1])

        Grafico2.dibujar_grafico(nodosvisitados,finales)


    def heuristica (self,x,y,j,k):
        return abs(x - j) + abs(y - k) #Manhattan

    def CalcObstaculos(self):

        Obstaculos = []
        veccolumnas = []
        vecfilas = [2,4,6,8,10]
        veccolumnas = [1,2,3,4,6,7,8,9,10,11]
        for j in vecfilas:
            for i in veccolumnas:
                Obstaculos.append((j,i))
        print(Obstaculos)
        return Obstaculos

AEstrella1 = Aestrella(5,2)
pares_inicio_final = [(1, 33), (2, 34), (3, 35),(15,22),(31,5),(5,19),(6,2),(12,41),(45,47),(31,43)]
for inicio, final in pares_inicio_final:
    AEstrella1.GraficarCamino([inicio, final])