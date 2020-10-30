import numpy as np

class Tracker:
    def __init__(self,A=1,H =1,R=0.1**2,Q=1e-6,P0=1, X0=0,dim=0,style='solid',start_y=0):
        if dim == 0: raise ValueError("the size of dim is 0!")
        # if len(R) != dim : raise ValueError("the dim of R is not initialized correctly!")
        # if len(Q) != dim : raise ValueError("the dim of Q is not initialized correctly!")
        # if len(P) != dim : raise ValueError("the dim of P is not initialized correctly!")
        # if len(X) != dim : raise ValueError("the dim of X is not initialized correctly!")
        self.A = A
        self.H = H
        self.R = R 
        self.Q = Q 
        self.P = P0
        self.X = X0
        self.dim = dim
        self.ttl = 1
        self.start = start_y
        # self.end = end
        self.style_count = [1,0] if style == 'solid' else [0,1]
        self.latest_style = style
        self.pred_count =0
        self.cross_count =0
    
    def update(self,meas,style,start):
        self.pred_count =0
        self.latest_style = style
        if style == 'solid':
            self.style_count[0]+=1
        else:
            self.style_count[1]+=1
        if start < self.start:
            self.start = start
        # self.end = end
        # prediction
        # assert(self.X.shape==(4,1))
        Xe = np.dot(self.A,self.X) 
        Pe = np.dot(np.dot(self.A,self.P), self.A.transpose())+ self.Q

        # measurment
        # print(self.H,self.P)
        # F = self.H * self.P
        M = np.dot(np.dot(self.H , self.P), self.H.transpose()) + self.R
        M = np.matrix(M)
        K = np.dot(np.dot(Pe , self.H.transpose()), M.I)
        # print('H * Xe shape is {}'.format(np.dot(self.H,Xe).shape))
        self.X = Xe + np.dot(K,meas - np.dot(self.H,Xe))
        self.X[0] = meas[0]
        self.X[1] = meas[1]
        # assert(self.X.shape==(4,1))
        self.P = np.dot(1-np.dot(K, self.H),Pe)

        # self.H = np.array([[1,0,0,0,0,0],\
        #     [0,-self.X[0],1,0,0,0]],dtype=np.float64)

        self.ttl +=1
        # self.X = meas
    def predict(self):
        Xe = np.dot(self.A,self.X) 
        self.X[:2] = Xe[:2]
        self.pred_count +=1
        
    def get_lane_style(self):
        history_style = 'solid' if self.style_count[0] > self.style_count[1]/3\
                        else 'dash'
        return history_style


def init_lane_tracker(X0,line_style,start_y):
    dim = 4
    d_t = 0.0667 
    A = np.array([[1,0,d_t,0],\
                    [0,1,0,d_t],\
                    [0,0,0,0],\
                    [0,0,0,0]],dtype=np.float64)
    H = np.array([[1,0,0,0],\
                [0,1,0,0]],dtype=np.float64)
    R = np.ones((2,2))*0.1**2
    Q = np.ones((dim,dim))*1e-6
    P0 = np.eye(dim)
    return Tracker(A,H,R,Q,P0,X0,dim,line_style,start_y)