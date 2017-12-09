import matplotlib.pylab as plt
import numpy as np
import numpy.random as r 

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def readData():
	X_scale = StandardScaler()
	digits = load_digits()
	y = digits.target
	X = X_scale.fit_transform(digits.data)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	y_train = makeVec(y_train)
	# y_test = makeVec(y_test)
	return X_train,X_test,y_train,y_test
	# print(X_train.shape , X_test.shape, y_train.shape, y_test.shape)
	# plt.gray()
	# plt.matshow(np.reshape(X[1],(8,8)))
	# plt.show() 	

def makeVec(y):
	yv = np.zeros((len(y),10))
	for i in xrange(len(y)):
		yv[i,y[i]] = 1
	return yv

def f(x):
	return 1/(1+np.exp(-x))

def f_deriv(x):
	return f(x)*(1-f(x))

def initializeW():
	W = {}
	b = {}
	for i in xrange(len(nn_structure)-1):
		W[i]=r.random_sample((nn_structure[i+1],nn_structure[i]))
		b[i]=r.random_sample((nn_structure[i+1],))
	return W,b

def initializedelta():
	W = {}
	b = {}
	for i in xrange(len(nn_structure)-1):
		W[i]=np.zeros((nn_structure[i+1],nn_structure[i]))
		b[i]=np.zeros((nn_structure[i+1],))
	return W,b

def forwardPass(x,w,b):
	h = {0:x}
	z = {}
	node_in = x 
	for i in xrange(len(w)):
		z[i+1] = w[i].dot(node_in)+b[i]
		h[i+1] = f(z[i+1])
		node_in = h[i+1]
	return h,z

def outDelta(y,h_out,z_out):
	return -(y-h_out)*f_deriv(z_out)

def hidDelta(nexDel,w_l,z_l):
	return np.dot(np.transpose(w_l),nexDel) * f_deriv(z_l)

def backProp(y,w,h,z,cost):
	delta = {}
	l = len(nn_structure)-1
	delta[l] = outDelta(y,h[l],z[l])
	cost += np.linalg.norm((y-h[l]))
	# print(avg_cost)
	# print(delta)
	for i in xrange(l-1,-1,-1):
		if(i>=1):
			delta[i] = hidDelta(delta[i+1],w[i],z[i])
		dw[i] += np.dot(delta[i+1][:,np.newaxis], np.transpose(h[i][:,np.newaxis])) 
		db[i] += delta[i+1]
	return cost

def predict_y(W, b, X, n_layers):
	m = X.shape[0]
	y = np.zeros((m,))
	for i in range(m):
		h, z = forwardPass(X[i, :], W, b)
		y[i] = np.argmax(h[n_layers-1])
		# print(y[i])
	return y


X,Xval,y,yval = readData()
nn_structure = [64, 30, 10]
w,b = initializeW()
# print(dw[0].shape)
nIter,alpha = 3000,0.25
print(X.shape)
m = len(y)
avg_cost_func = []
while(nIter):
	dw,db = initializedelta()
	avg_cost = 0
	for i,x in enumerate(X) :
		h,z = forwardPass(x,w,b)
		avg_cost = backProp(y[i],w,h,z,avg_cost)
	for i in xrange(len(nn_structure)-1):
		w[i] += -alpha * (1.0/m * dw[i])
		b[i] += -alpha * (1.0/m * db[i])
	avg_cost = avg_cost/m
	avg_cost_func.append(avg_cost)
	nIter-=1

plt.plot(avg_cost_func)
plt.show()

y_pred = predict_y(w, b, Xval, 3)
acc = accuracy_score(yval, y_pred)*100
print(acc)