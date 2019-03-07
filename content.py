
import numpy as np
from utils import polynomial

def mean_squared_error(x, y, w):
    '''
    :param x: input vector Nx1
    :param y: output vector Nx1
    :param w: model parameters (M+1)x1
    :return: mean squared error between output y
    and model prediction for input x and parameters w
    '''

    #matrix of predicted result
    #pred= polynomial(x,w)

    #calculating the size of matrix y
    size=y.shape[0]

    #using the formula for mean square error

    R=0

    #loop for summing
    for i in range (0,size):
        add=((y[i,0]-polynomial(x[i,0],w))**2)
        R=R+add

    #division by size
    R=R/size
    X=R[0].astype(float)

    return (X)


def design_matrix(x_train,M):
    '''
    :param x_train: input vector Nx1
    :param M: polynomial degree 0,1,2,...
    :return: Design Matrix Nx(M+1) for M degree polynomial
    '''

    # calculating the size of matrix x
    size = x_train.shape[0]

    # creating design matrix and initializing to 0

    #design=np.zeros((size,(M+1))
    design = np.zeros((size,M+1))

    #creating elements of design matrix

    #loop for numbers of rows
    for i in range(0, size):
        for j in range (0,(M+1)):
            design[i, j] = x_train[i, 0] ** j

    return design





def least_squares(x_train, y_train, M):
    '''
    :param x_train: training input vector  Nx1
    :param y_train: training output vector Nx1
    :param M: polynomial degree
    :return: tuple (w,err), where w are model parameters and err mean squared error of fitted polynomial
    '''

    #find design matrix for x
    design = design_matrix(x_train,M)

    #create matrix of parameters w and intialize it to 0
    w=np.zeros((M,1))

    #find values of w
    w=(np.linalg.inv(design.transpose()@design)@design.transpose()@y_train)

    #find EMS
    err=mean_squared_error(x_train, y_train, w)


    return (w,err)



def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    '''
    :param x_train: training input vector Nx1
    :param y_train: training output vector Nx1
    :param M: polynomial degree
    :param regularization_lambda: regularization parameter
    :return: tuple (w,err), where w are model parameters and err mean squared error of fitted polynomial with l2 regularization
    '''

    # find design matrix for x
    design = design_matrix(x_train, M)

    # create matrix of parameters w and intialize it to 0
    w = np.zeros((M, 1))

    #create identity matrix of given size
    size=design.shape[1]
    I=np.identity(size)


    # find values of w
    w = (np.linalg.inv((design.transpose() @ design) + (regularization_lambda * I)) @ design.transpose() @ y_train)

    # find EMS
    err = mean_squared_error(x_train, y_train, w)

    # create a tuple to return
    values = (w, err)

    return values



def model_selection(x_train, y_train, x_val, y_val, M_values):
    '''
    :param x_train: training input vector Nx1
    :param y_train: training output vector Nx1
    :param x_val: validation input vector Nx1
    :param y_val: validation output vector Nx1
    :param M_values: array of polynomial degrees that are going to be tested in model selection procedure
    :return: tuple (w,train_err, val_err) representing model with the lowest validation error
    w: model parameters, train_err, val_err: training and validation mean squared error
    '''

    # finding size of array M_values
    size = len(M_values)

    #create matrix of results to compare M w val_error
    result=np.zeros((size,2))


    for i in range (0,size):
        result[i,0]=M_values[i]
        result[i,1]= mean_squared_error(x_val,y_val,least_squares(x_train,y_train,M_values[i])[0])


    #finding minimum
    minim=np.amin(result, axis=0)[1]


    for i in range (0,size):
        if(result[i,1]==minim):
            best_M_pos=i


    return (least_squares(x_train,y_train,M_values[best_M_pos])[0],least_squares(x_train,y_train,M_values[best_M_pos])[1], minim)



def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    '''
    :param x_train: training input vector Nx1
    :param y_train: training output vector Nx1
    :param x_val: validation input vector Nx1
    :param y_val: validation output vector Nx1
    :param M: polynomial degree
    :param lambda_values: array of regularization coefficients are going to be tested in model selection procedurei
    :return:  tuple (w,train_err, val_err, regularization_lambda) representing model with the lowest validation error
    (w: model parameters, train_err, val_err: training and validation mean squared error, regularization_lambda: the best value of regularization coefficient)
    '''
    # finding size of array M_values
    size = len(lambda_values)

    # create matrix of results to compare M w val_error
    result = np.zeros((size, 2))

    for i in range(0, size):
        result[i, 0] = lambda_values[i]
        result[i, 1] = mean_squared_error(x_val, y_val, regularized_least_squares(x_train,y_train,M,lambda_values[i])[0])


    # finding minimum
    minim = np.amin(result, axis=0)[1]



    for i in range(0, size):
        if (result[i, 1] == minim):
            best_lambda_pos = i

    return (regularized_least_squares(x_train,y_train,M,lambda_values[best_lambda_pos])[0],regularized_least_squares(x_train,y_train,M,lambda_values[best_lambda_pos])[1],minim,lambda_values[best_lambda_pos])


# CHECK MAIN


'''x=np.matrix([[37],[20],[36],[45]])
x_v=np.matrix([[55],[47],[100],[80]])
y=np.matrix([[320],[150],[252],[297]])
y_v=np.matrix([[352],[300],[450],[420]])
w=np.matrix([[9.7481],[8.4887],[-0.0412]])
MM=np.array([1,2,3,4,5,6,7])

#print(MM[0])
#print(x)
#print(design_matrix(x,2))
#print(regularized_least_squares(x,y,2,0.6))
#print(least_squares(x,y,5))
#print(model_selection(x, y, x_v, y_v, MM))
#print(regularized_model_selection(x,y,x_v,y_v,6,MM))
#print(polynomial(70,w))
#print(mean_squared_error(x,y,w))'''