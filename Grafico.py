import tkinter as tk

class Grafico:
    def __init__(self, filas, columnas, obstaculos):

        self.filas = filas+1
        self.columnas = columnas+1
        self.obstaculos = obstaculos

    def dibujar_grafico(self,camino,final):
        self.camino = camino
        self.final = final
        print(self.camino)

        matriz = [[0 for j in range(self.columnas-1)] for i in range(self.filas-1)] #Matriz de ceros de 10x10 (filas x columnas)

        for i, j in self.obstaculos:
            matriz[i][j] = 2 #Obstaculos
        r=4
        for sublist in self.camino:
            for subsublist in sublist:
                i, j = subsublist[0], subsublist[1]
                matriz[i][j] = r #Camino
            r+=1
        for i, j in self.final:
            matriz[i][j] = 3 #Final

        for i in range(self.filas-1):
            for j in range(self.columnas-1):
                if matriz[i][j] == 0:
                    matriz[i][j] = 1 #Espacio vacio
        ventana = tk.Tk()
        ventana.title("Grafico")

        lienzo = tk.Canvas(ventana)

        for i in range(self.filas-1):
            for j in range(self.columnas-1):
                x1 = j * 30
                y1 = i * 30
                x2 = x1 + 30
                y2 = y1 + 30
                if matriz[i][j] == 1:
                    color = "white" 
                if matriz[i][j] == 2:
                    color = "black"
                elif matriz[i][j] == 3:
                    color = "red"
                elif matriz[i][j] == 4:
                    color = "yellow"
                lienzo.create_rectangle(x1, y1, x2, y2, fill=color)

        lienzo.config(width=self.columnas * 30, height=self.filas * 30)

        lienzo.pack()
        ventana.mainloop()