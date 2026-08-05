import numpy as np

def calc_mape_stats(y_true, y_pred, metrics):
    mape=0
    mape_sq=0
    best_mape=np.finfo(float).max
    worst_mape=0
    mape_matrix=np.abs((y_pred-y_true)/y_pred)
    for i in range(y_true.shape[0]):
        mape_i=0
        cnt=0
        for j in range(y_true.shape[1]):
            if y_true[i,j]>0.1 and y_pred[i,j]>0.1:
                mape_i=mape_i+mape_matrix[i,j]
                cnt=cnt+1
        if cnt==0:
            mape_i=np.nan
        best_mape=min(best_mape,mape_i)
        worst_mape=max(worst_mape,mape_i)
        mape=mape+mape_i/cnt
        mape_sq=mape_sq+mape_i**2/cnt
    mape=mape/y_true.shape[0]
    mape_sq=mape_sq/y_true.shape[0]
    metrics['MAPE'] = mape
    metrics['MAPE std'] = np.sqrt(mape_sq-mape**2)
    metrics['Worst MAPE'] = worst_mape
    metrics['Best MAPE'] = best_mape

def calc_mre(y_true, y_pred, metrics):
    mre=[]
    mre_matrix=np.abs((y_pred-y_true)/y_pred)
    for j in range(y_true.shape[1]):
        mre_j=0
        cnt=0
        for i in range(y_true.shape[0]):
            if y_true[i,j]>0.1 and y_pred[i,j]>0.1:
                mre_j=mre_j+mre_matrix[i,j]
                cnt=cnt+1
        if cnt:
            mre.append(mre_j/cnt)
        else:
            mre.append(np.nan)
    metrics['MRE'] = np.array(mre)

def calc_mase_stats(y_true, y_pred, y_train, metrics):
    mase=np.mean(np.abs(y_true-y_pred),axis=1)/np.mean(np.abs((y_true-np.mean(y_train,axis=0))),axis=1)
    mase_sq=np.mean(mase**2)
    mase_mean=np.mean(mase)
    metrics['MASE'] = mase_mean
    metrics['MASE std'] = np.sqrt(mase_sq-mase_mean**2)
    metrics['Worst MASE'] = np.max(mase)
    metrics['Best MASE'] = np.min(mase)

def calc_mrcpe_stats(y_true, y_pred, metrics):
    # Calculate MRCPE (Mean Relative Cumulative Percentage Error)
    mrcpe= np.mean(np.mean(np.sum(np.abs(y_true - y_pred), axis=1) / np.sum(y_true, axis=1) * 100))
    mrcpe_sq=np.mean(mrcpe**2)
    mrcpe_mean=np.mean(mrcpe)
    metrics['MRCPE'] = mrcpe_mean
    metrics['MRCPE std'] = np.sqrt(mrcpe_sq-mrcpe_mean**2)
    metrics['Worst MRCPE'] = np.max(mrcpe)
    metrics['Best MRCPE'] = np.min(mrcpe)