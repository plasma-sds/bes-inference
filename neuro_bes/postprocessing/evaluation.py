import numpy as np

def calc_mape_stats(y_true, y_pred, metrics):
    eps=1e-8
    mape_matrix=np.abs((y_pred-y_true)/(y_pred+eps))
    per_row_mape=np.empty(y_true.shape[0])
    for i in range(y_true.shape[0]):
        mape_i=0
        cnt=0
        for j in range(y_true.shape[1]):
            if y_true[i,j]>0.1 and y_pred[i,j]>0.1:
                mape_i=mape_i+mape_matrix[i,j]
                cnt=cnt+1
        per_row_mape[i]=mape_i/cnt if cnt>0 else np.nan
    # rows with no point above the 0.1 threshold for either true or predicted are excluded via
    # nan-aware aggregation, rather than letting a single such row poison the whole mean to nan
    mape=np.nanmean(per_row_mape)
    mape_sq=np.nanmean(per_row_mape**2)
    metrics['MAPE'] = mape
    metrics['MAPE std'] = np.sqrt(mape_sq-mape**2)
    metrics['Worst MAPE'] = np.nanmax(per_row_mape)
    metrics['Best MAPE'] = np.nanmin(per_row_mape)
    metrics['MAPE per profile'] = per_row_mape

def calc_mre(y_true, y_pred, metrics):
    eps=1e-8
    mre=[]
    mre_matrix=np.abs((y_pred-y_true)/(y_pred+eps))
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

def calc_rmse_stats(y_true, y_pred, metrics):
    rmse=np.sqrt(np.mean((y_true-y_pred)**2,axis=1))
    rmse_sq=np.mean(rmse**2)
    rmse_mean=np.mean(rmse)
    metrics['RMSE'] = rmse_mean
    metrics['RMSE std'] = np.sqrt(rmse_sq-rmse_mean**2)
    metrics['Worst RMSE'] = np.max(rmse)
    metrics['Best RMSE'] = np.min(rmse)

def calc_mrcpe_stats(y_true, y_pred, metrics):
    # MRCPE (Mean Relative Cumulative Percentage Error): per profile, the ratio of total absolute
    # error to total true signal (a ratio of sums, not an average of point-wise ratios like MAPE).
    per_profile_mrcpe=np.sum(np.abs(y_true - y_pred), axis=1) / np.sum(y_true, axis=1) * 100
    mrcpe_mean=np.mean(per_profile_mrcpe)
    mrcpe_sq=np.mean(per_profile_mrcpe**2)
    metrics['MRCPE'] = mrcpe_mean
    metrics['MRCPE std'] = np.sqrt(mrcpe_sq-mrcpe_mean**2)
    metrics['Worst MRCPE'] = np.max(per_profile_mrcpe)
    metrics['Best MRCPE'] = np.min(per_profile_mrcpe)
    metrics['MRCPE per profile'] = per_profile_mrcpe