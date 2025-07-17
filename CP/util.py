import numpy as np
import matplotlib.pyplot as plt

def regression_plot_with_uncertainty(
    ax,
    prediction_inputs,
    mean_prediction,
    prediction_bounds,
    scatter_points,
    quantiles=[1.0],
    labels=['Estimated Error'] #估计误差
):
    """
    绘制包含均值预测线和预测区间阴影区域的图像。
    如果提供了多个分位数，则按照从最大分位数开始的顺序依次绘制阴影区域。
    散点数据将单独绘制。

    参数情况
    ----------
    ax : matplotlib subplot axis，子图轴对象
    prediction_inputs : NDArray，形状为(n_samples, 1)
        用于预测均值和预测区间的位置
    mean_prediction : NDArray，形状为(n_samples, 1)
        预测的均值（标准模型预测）
    prediction_bounds : NDArray，形状为(n_samples, 2, n_quantiles)
        各个分位数对应的预测区间的上下界
    scatter_points : Tuple[NDArray of shape (n_scatter_samples, 1), NDArray of shape (n_scatter_samples, 1)]
        需要在图像上额外绘制的输入数据/对应标签
    quantiles : list
        包含不同预测区间对应的分位数
    labels : list
        包含每个分位数对应的标签，这些标签将添加到图例中
    """
    
    order = np.argsort(prediction_inputs[:, 0], axis=0)

    cmap = plt.cm.Oranges
    indices = np.linspace(0.2, 0.4, num=len(quantiles))
    colors = cmap(indices)

    # We reverse the quantiles because we want to plot the largest bounds first
    quantile_indices = np.argsort(quantiles)[::-1]
    for i in quantile_indices:
        ax.fill_between(
            prediction_inputs[order].ravel(),
            prediction_bounds[order][:, 0, i].ravel(),
            prediction_bounds[order][:, 1, i].ravel(),
            alpha=1.0,
            color=colors[i],
            label=labels[i]
        )

    ax.scatter(scatter_points[0], scatter_points[1], color='blue', s=5)
    ax.plot(prediction_inputs[order], mean_prediction[order], color='red', label='Prediction')
    ax.legend()

def regression_plot_with_scores(
    inputs,
    mean_prediction,
    prediction_bands,
    scatter_points,
    train_scores,
    cal_scores,
    test_scores,
    separators=[0, 1.0],
    quantile=0.95,
    labels=[f'95% Target Coverage']
):
    """ Plot the regression fit with prediction intervals (on the left) and score distributions (on the right). """
    
    fig, ax = plt.subplot_mosaic([
        ['left', 'upper right'],
        ['left', 'center right'],
        ['left', 'lower right']
    ], figsize=(12, 4), layout="constrained")
    
    regression_plot_with_uncertainty(
        ax['left'],
        inputs,
        mean_prediction,
        prediction_bands,
        scatter_points,
        quantiles=[quantile],
        labels=labels
    )
    ax['left'].set_title(r'Predictive Bands after Calibration: $\hat{\mu}(X) \pm q$' + '\nData: Train | Calibration | Test')
    ax['left'].vlines(separators[1:-1], ymin=prediction_bands[:, 0].min(), ymax=prediction_bands[:, 1].max(), colors='black')
    ax['left'].set_xlim(separators[0], separators[-1])
    ax['upper right'].hist(train_scores, bins=50)
    ax['upper right'].set_title('Training Scores (Coverage: {:.2f}%)'.format((train_scores < quantile).mean()*100))
    ax['upper right'].sharex(ax['lower right'])
    ax['center right'].hist(cal_scores, bins=50)
    ax['center right'].set_title('Calibration Scores (Coverage: {:.2f}%)'.format((cal_scores < quantile).mean()*100))
    ax['center right'].sharex(ax['lower right'])
    ax['lower right'].hist(test_scores, bins=50)
    ax['lower right'].set_title('Test Scores (Coverage: {:.2f}%)'.format((test_scores < quantile).mean()*100))
    
    plt.show()

def compute_coverage(y, prediction_bands):
    """ Computes the coverage of the prediction bands on the data y. """
    num_points_outside = np.sum((y < prediction_bands[:,0]) | (y > prediction_bands[:,1]))
    coverage_score = 1.0-num_points_outside/len(y)
    return coverage_score