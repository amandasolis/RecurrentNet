import binascii
import numpy as np
#from numpy import *
#from pylab import *
#from struct import *
import math
from random import sample
from random import shuffle
import random
import os, struct
from array import array as pyarray
#from scipy import stats


data = open('input1.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
N = len(data)
numberCharacters = len(chars)
print 'data has %d characters, %d unique.' % (N, numberCharacters)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
outfile = open('out.txt', 'w')

def prob(a):
	exparray = np.frompyfunc(math.exp, 1, 1)
	exp = exparray(a)
	sumexp = np.sum(exp)
	return exp / sumexp

def probWT(a, temperature):
	exparray = np.frompyfunc(math.exp, 1, 1)
	exp = exparray(a/temperature) 
	sumexp = np.sum(exp)
	return exp / sumexp

""" input is a matrix of the inputs to the hidden layer at many timesteps"""
def hiddenFunction(a):
	return np.tanh(a)

def hiddenLayerDerivative(a):
	return 1 - (a ** 2)

numHiddenUnits = 200
learningRate = 0.1
sequenceLength = 25

"""Returns a tuple of the activiations for the hidden and output layer"""
def forwardProp(x, y, Wxh, Whh, Why, b1, b2, hiddenLast):
	loss = 0
	h = np.zeros((sequenceLength+1, numHiddenUnits))
	h[-1] = hiddenLast
	z = np.zeros((sequenceLength, numberCharacters))
	for t in range(0, sequenceLength):	
		a1 = np.dot(x[t], np.transpose(Wxh))
		a2 = np.dot(h[t-1], np.transpose(Whh))
		h[t] = hiddenFunction(a1 + a2 + b1)
		z[t] = prob(np.dot(h[t], np.transpose(Why)) + b2)
		# print(z[t][np.argmax(y[t])])
		loss += -np.log(z[t][np.argmax(y[t])])
	return loss, h, z

mWxh, mWhh, mWhy = np.zeros((numHiddenUnits, numberCharacters)), np.zeros((numHiddenUnits, numHiddenUnits)), np.zeros((numberCharacters, numHiddenUnits))
mbh, mby = np.zeros(numHiddenUnits), np.zeros(numberCharacters) # memory variables for Adagrad


def train(x, y, Wxh, Whh, Why, b1, b2, hiddenLast):
	loss, h, z = forwardProp(x, y, Wxh, Whh, Why, b1, b2, hiddenLast)
	changeWhy = np.zeros((numberCharacters, numHiddenUnits))
	changeWhh = np.zeros((numHiddenUnits, numHiddenUnits))
	changeWxh = np.zeros((numHiddenUnits, numberCharacters))
	changeb1 = np.zeros(numHiddenUnits)
	changeb2 = np.zeros(numberCharacters)

	for t in range(0, sequenceLength):
		delta2 = np.subtract(z[t], y[t])
		changeWhy += np.outer(delta2, h[t])
		changeb2 += delta2
		delta1 = np.multiply(hiddenLayerDerivative(h[t]), np.dot(np.transpose(Why), delta2))
		changeb1 += delta1
		changeWhh += np.outer(delta1, h[t - 1])
		changeWxh += np.outer(delta1, x[t])
	for dparam in [changeWhy, changeWhh, changeWxh, changeb1, changeb2]:
		np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

	# perform parameter update with Adagrad
	for param, dparam, mem in zip([Wxh, Whh, Why, b1, b2], 
                                [changeWxh, changeWhh, changeWhy, changeb1, changeb2], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
		mem += dparam * dparam
		param += -learningRate * dparam / np.sqrt(mem + 1e-8) # adagrad update

	return loss, h[sequenceLength-1]

def genText(startChar, Wxh, Whh, Why, b1, b2, temperate=1.0):
	results = []
	textLength = 100
	inArry = startChar
	h = [0] * numHiddenUnits
	for t in range(0, textLength):
		a1 = np.dot(inArry, np.transpose(Wxh))
		a2 = np.dot(h, np.transpose(Whh))
		h = hiddenFunction(a1 + a2 + b1)
		p = probWT(np.dot(h, np.transpose(Why)) + b2, temperate).astype(float)
		characterIndex = np.random.choice(range(numberCharacters), p=p.ravel())
		inArry = np.zeros(numberCharacters)
		inArry[characterIndex] = 1
		results.append(ix_to_char[characterIndex])
	return results

np.random.seed(0)
Wxh = np.random.rand(numHiddenUnits, numberCharacters)
Whh = np.random.rand(numHiddenUnits, numHiddenUnits)
Why = np.random.rand(numberCharacters, numHiddenUnits)
maxValueFirst = math.sqrt(12) * (1/math.sqrt(numberCharacters))
maxValueSecond = math.sqrt(12) * (1/math.sqrt(numHiddenUnits))
Wxh = Wxh * maxValueFirst
Whh = Whh * maxValueSecond
Why = Why * maxValueSecond
Wxh = Wxh - maxValueFirst/2
Whh = Whh - maxValueSecond/2
Why = Why - maxValueSecond / 2
b1 = np.zeros(numHiddenUnits)
b2 = np.zeros(numberCharacters)

smooth_loss = -np.log(1.0/numberCharacters)*sequenceLength # loss at iteration 0


X = np.zeros((N-1, numberCharacters))
Y = np.zeros((N-1, numberCharacters))

for i in range(0, N):
	if i < N-1:
	    X[i][char_to_ix[data[i]]] = 1
	if i > 0:
		Y[i-1][char_to_ix[data[i]]] = 1


N = N - N%sequenceLength

# 50 epochs
iteration = 0
losses = []
for j in range(0, 50):
	for i in xrange(0, N, sequenceLength):
		x = X[i:i+sequenceLength]
		y = Y[i:i+sequenceLength]
		if iteration == 0:
			hiddenLast = np.zeros(numHiddenUnits)
		loss, hiddenLast = train(x, y, Wxh, Whh, Why, b1, b2, hiddenLast)
		smooth_loss = smooth_loss * 0.999 + loss * 0.001
		iteration += 1

		"""if iteration == 20000:
			for temperature in [0.1, 1.1, 10]:
				for k in range(0, 5):
					output = ''.join(genText(X[i+k], Wxh, Whh, Why, b1, b2, temperature))
					print('Temperature is: %f ----\n %s \n----' % (temperature, output))"""
		if iteration % 100 == 0:
			#print 'iter %d, loss=%f' % (iteration, smooth_loss)
			losses.append(smooth_loss)
		if iteration ==100 or iteration == 1000 or iteration == 10000 or iteration == 20000 or iteration == 60000:
			output = ''.join(genText(X[i], Wxh, Whh, Why, b1, b2))
			print 'iteration %d ----\n %s \n----' % (iteration, output)
		if iteration == 60000:
			print losses


#Wxh = np.zeros()

