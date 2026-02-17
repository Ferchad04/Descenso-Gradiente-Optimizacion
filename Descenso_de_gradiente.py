import numpy as np
import matplotlib.pyplot as plt


# Clase abstracta para funciones objetivo
class Funcion:
    def eval(self, x):
        pass
    def diff(self, x):
        pass
    def hess(self, x):
        pass
    def plot(self, xmin, xmax):
        x = np.linspace(xmin, xmax, 100)
        y = np.linspace(xmin, xmax, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.eval(np.array([X[i, j], Y[i, j]]))
                
        plt.figure(figsize=(8, 6))
        cp = plt.contourf(X, Y, Z, levels=20, cmap='jet')
      
        plt.contour(X, Y, Z, levels=20, colors='black', linewidths=0.5, alpha=0.7)
        plt.colorbar(label='Valor de f(x)')
        plt.title(f"Contorno de {self.__class__.__name__}")
        plt.xlabel('x1')
        plt.ylabel('x2')


class Esfera(Funcion):
    def eval(self, x):
        return np.sum(x**2)
    
    def diff(self, x):
        return 2 * x
    
    def hess(self, x):
        return 2 * np.eye(len(x))

class Rosenbrock(Funcion):
    def eval(self, x):
        
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    
    def diff(self, x):
        grad = np.zeros_like(x)
        x0, x1 = x[0], x[1]
        grad[0] = -400 * x0 * (x1 - x0**2) - 2 * (1 - x0)
        grad[1] = 200 * (x1 - x0**2)
        return grad
    
    def hess(self, x):
        hess = np.zeros((2, 2))
        x0, x1 = x[0], x[1]
        hess[0, 0] = 1200 * x0**2 - 400 * x1 + 2
        hess[0, 1] = -400 * x0
        hess[1, 0] = -400 * x0
        hess[1, 1] = 200
        return hess

class Cigarro(Funcion):
    def eval(self, x):
        return x[0]**2 + 1e6 * np.sum(x[1:]**2)
    
    def diff(self, x):
        grad = np.zeros_like(x)
        grad[0] = 2 * x[0]
        grad[1:] = 2 * 1e6 * x[1:]
        return grad
    
    def hess(self, x):
        n = len(x)
        hess = np.zeros((n, n))
        hess[0, 0] = 2
        for i in range(1, n):
            hess[i, i] = 2 * 1e6
        return hess
    

class DescensoGradiente:
    def __init__(self, funcion_obj, tam_paso, max_iter=1000, tol=1e-6):
        self.funcion = funcion_obj
        self.tam_paso = tam_paso
        self.max_iter = max_iter
        self.tol = tol
        self.camino = [] 

    def solve(self, x0):
        x = np.array(x0, dtype=float)
        self.camino = [x.copy()]
        
        for i in range(self.max_iter):
            grad = self.funcion.diff(x)
            
            # Paso de gradiente
            x_new = x - self.tam_paso * grad
            self.camino.append(x_new.copy())
            
            if np.linalg.norm(grad) < self.tol:
                print(f"Convergencia en iteración {i}")
                break
            x = x_new
        return x

    def plot2d(self, xmin, xmax):

        self.funcion.plot(xmin, xmax)
        
        ruta = np.array(self.camino)
        plt.plot(ruta[:, 0], ruta[:, 1], 'r.-', label='Trayectoria', linewidth=1)
        plt.plot(ruta[0, 0], ruta[0, 1], 'bo', label='Inicio')
        plt.plot(ruta[-1, 0], ruta[-1, 1], 'rx', label='Fin', markersize=10)
        plt.legend()
        plt.show()


def main():
    print("Esfera")
    f = Esfera()
    gd = DescensoGradiente(f, tam_paso=0.05, max_iter=1000)
    gd.solve([4.0, 3.0])
    gd.plot2d(-5, 5)

    print("\nRosenbrock")
    f = Rosenbrock()
    gd = DescensoGradiente(f, tam_paso=0.0015, max_iter=5000)
    gd.solve([-1.5, 2.0])
    gd.plot2d(-2, 3)

    print("\nCigarro")
    f = Cigarro()
    gd = DescensoGradiente(f, tam_paso=1e-7, max_iter=2000)
    gd.solve([2.0, 2.0])
    gd.plot2d(-3, 3)

if __name__ == "__main__":
    main()