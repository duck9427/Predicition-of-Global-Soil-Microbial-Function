# 导入第三方库
import numpy as np  # 科学计算库
import matplotlib.pyplot as plt  # 数据可视化库
import pandas as pd  # 数据处理库
import seaborn as sns  # 高级数据可视化库
from sklearn.model_selection import train_test_split, cross_val_score  # 导入数据集拆分工具和交叉验证工具
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score  # 模型评估方法
import sklearn.metrics as metrics  # 导入模型评估工具
from sklearn.ensemble import RandomForestRegressor  # 随机森林回归模型

# 定义相关系数R2
def evaluate_model(model,x,y):
    pred = model.predict(x)
    r2 = metrics.r2_score(y,pred)
    corr_coef = np.corrcoef(y, pred)[0, 1]
    return r2, corr_coef

# 定义计算粒子的距离
def getEuclidean(vec1, vec2):
    dist = vec1 - vec2  # 相减
    dist = dist ** 2  # 平方
    dist = dist[0] + dist[1]  # 相加
    dist = dist ** 1 / 2  # 平方
    return dist  # 返回距离数据


# 定义适应度计算函数
def getFitness(X_train, X_test, y_train, y_test, x):
    fitness_list = []  # 定义一个空列表

    if abs(x[0][0][0]) > 0:  # 判断取值
        max_features = int(abs(x[0][0][0])) + 6  # 赋值
    else:
        max_features = 6  # 等于0的话  赋值为常量

    if abs(x[0][0][1]) > 0:  # 判断取值
        n_estimators = int(abs(x[0][0][1])) + 650  # 赋值
    else:
        n_estimators = 650  # 等于0的话  赋值为常量
    rfc_model = RandomForestRegressor(max_features=max_features, n_estimators=n_estimators)  # 建模
    rfc_model.fit(X_train, y_train)  # 拟合
    cv_accuracies = cross_val_score(rfc_model, X_test, y_test, cv=3, scoring='r2')  # 交叉验证计算r方
    # 使错误率降到最低
    accuracies = cv_accuracies.mean()  # 取交叉验证均值
    fitness_value = 1 - accuracies  # 错误率 赋值 适应度函数值

    fitness_list.append(fitness_value)  # 数据存入列表
    return np.array(fitness_list)  # 返回适应度数据


# 定义主函数
if __name__ == '__main__':

    # 读取数据
    df = pd.read_csv(r'C:\Users\Administrator\Desktop\LH\NData-NFM-HYQ.csv')

    # 用Pandas工具查看数据
    print(df.head())

    # 查看数据集摘要
    print(df.info())

    # 数据描述性统计分析
    print(df.describe())

    # y变量分布直方图
    fig = plt.figure(figsize=(8, 5))  # 设置画布大小
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    data_tmp = df['NFM']  # 过滤出y变量的样本
    # 绘制直方图  bins：控制直方图中的区间个数 auto为自动填充个数  color：指定柱子的填充色
    plt.hist(data_tmp, bins='auto', color='g')
    plt.xlabel('EP')
    plt.ylabel('Counts')
    plt.title('EP distribution')
    plt.show()

    # 数据的相关性分析

    sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)  # 绘制热力图
    plt.title('Correlation heatmap')
    plt.show()

    # 提取特征变量和标签变量
    y = df['NFM']
    X = df.drop('NFM', axis=1)
#    X = df.drop(['NPP', 'lat', 'lon'], axis=1)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 参数初始化 用于更新加速度系数
    # 个体认知加速度系数初始值
    c1i = 2.5
    c1f = 0.5
    # 社会认知加速度系数初始值
    c2i = 0.5
    c2f = 2.5

    # 分布式时延项的强度因子
    ml = 0
    mg = 0

    # 惯性权重
    W_MAX = 0.8  # 惯性权重最大值
    W_MIN = 0.3  # 惯性权重最下值
    w = W_MAX

    maxgen = 20  # 进化次数
    sizepop = 50  # 种群规模

    # 粒子速度和位置的范围
    Vmax = 0.1  # 粒子最大速度初始化
    Vmin = 0.05  # 粒子最小速度初始化

    data_points = X.values

    dataset = data_points  # 赋值

    popmax = np.max(dataset)  # 种群最大值
    popmin = np.min(dataset)  # 种群最小值

    # 初始化参数值
    K = 2

    # 产生初始粒子和速度
    pop = np.zeros([sizepop, 16, K])  # 初始化为0
    pop = np.random.uniform(popmin, popmax, size=[sizepop, 16, K])  # 进行随机初始化
    for i in range(sizepop):  # 循环
        pop[i] = np.array(
            [dataset[np.random.randint(0, len(dataset))], dataset[np.random.randint(0, len(dataset))]]).T  # 进行粒子位置赋值
    v = np.random.uniform(Vmin, Vmax, size=[sizepop, 16, K])  # 进行速度赋值


    # 计算初始时每个粒子到其他粒子的平均距离
    def dist(pop):
        d = np.zeros(sizepop)  # 距离初始化为0
        for pos in range(sizepop):  # 循环
            d_sum = 0  # 初始化为0
            for j in range(sizepop):  # 循环
                if pos == j:  # 判断是否为自己
                    continue
                temp = (pop[pos] - pop[j]) ** 2  # 生成距离临时数值
                dj_sum = 0  # 初始化为0
                for n1 in range(temp.shape[0]):  # 进行循环
                    for n2 in range(temp.shape[1]):  # 循环
                        dj_sum = dj_sum + temp[n1][n2]  # 距离计算
                d_sum += dj_sum ** 1 / 2  # 赋值
            d[pos] = d_sum / sizepop  # 粒子位置赋值
        return d  # 返回距离数据


    d = dist(pop)  # 初始时每个粒子到其他粒子的平均距离
    fitness = getFitness(X_train, X_test, y_train, y_test, pop)  # 计算适应度
    i = np.argmin(fitness)  # 找最好的个体
    Ef = (d[i] - np.min(d)) / (np.max(d) - np.min(d))  # 计算进化因子
    gbest = pop  # 记录个体最优位置
    gbest_history = [gbest]  # 记录个体最优位置历史值
    zbest = pop[i]  # 记录群体最优位置
    zbest_history = [zbest]  # 记录群体最优位置历史值
    fitnessgbest = fitness  # 个体最佳适应度值
    fitnesszbest = fitness[i]  # 全局最佳适应度值
    # 根据Ef寻找进化状态
    if 0 <= Ef < 0.25:  # 判断 收敛状态
        epsilonk = 1  # 赋值
    if 0.25 <= Ef < 0.5:  # 判断 开发状态
        epsilonk = 2  # 赋值
    if 0.5 <= Ef < 0.75:  # 判断 勘探状态
        epsilonk = 3  # 赋值
    if 0.75 <= Ef <= 1:  # 判断 跳出状态
        epsilonk = 4  # 赋值
    N = 200  # 分布式时延步数的上限值
    alpha = np.random.randint(2, size=N)  # 随机初始化数值
    evo_state = {1: (0, 0), 2: (0.01, 0), 3: (0, 0.01), 4: (0.01, 0.01)}  # 初始数值
    ml, mg = evo_state[epsilonk]  # 分布式时延项的强度因子赋值

    # 迭代寻优
    t = 0
    record_RODDPSO = np.zeros(maxgen)  # 初始化为0
    while t < maxgen:  # 循环

        print('************************************', 'This is the ', t + 1, 'th iteration***************************************')

        # 惯性权重更新
        w = W_MAX - ((W_MAX - W_MIN) * t / maxgen)

        # 加速度系数更新
        c1 = ((c1f - c1i) * (maxgen - t) / maxgen) + c1i  # 计算、赋值
        c2 = ((c2f - c2i) * (maxgen - t) / maxgen) + c2i  # 计算、赋值
        c3 = c1  # 赋值
        c4 = c2  # 赋值

        # 计算每个粒子的平均距离
        d = dist(pop)

        # 更新Ef
        Ef = (d[i] - np.min(d)) / (np.max(d) - np.min(d))

        # 根据Ef寻找进化状态
        if 0 <= Ef < 0.25:  # 判断 收敛状态
            epsilonk = 1  # 赋值
        if 0.25 <= Ef < 0.5:  # 判断 开发状态
            epsilonk = 2  # 赋值
        if 0.5 <= Ef < 0.75:  # 判断 勘探状态
            epsilonk = 3  # 赋值
        if 0.75 <= Ef <= 1:  # 判断 跳出状态
            epsilonk = 4  # 赋值

        ml, mg = evo_state[epsilonk]  # 分布式时延项的强度因子赋值

        # 速度更新

        mcr3 = 0  # 数值初始化
        mcr4 = 0  # 数值初始化
        for tao in range(N):  # 循环
            if t >= tao:  # 判断
                mcr3 += alpha[tao] * (gbest_history[t - tao] - pop)  # 随机向量赋值
                mcr4 += alpha[tao] * (zbest_history[t - tao] - pop)  # 随机向量赋值
            else:
                mcr3 += alpha[tao] * (gbest_history[t] - pop)  # 随机向量赋值
                mcr4 += alpha[tao] * (zbest_history[t] - pop)  # 随机向量赋值

        # 速度计算
        v = w * v + c1 * np.random.random() * (gbest - pop) + c2 * np.random.random() * (zbest - pop) \
            + ml * c3 * np.random.random() * mcr3 + mg * c4 * np.random.random() * mcr4
        v[v > Vmax] = Vmax  # 速度大于最大值  赋值为最大值
        v[v < Vmin] = Vmin  # 速度小于最小值 赋值为最小值

        # 位置更新
        pop = pop + v  # 计算位置
        pop[pop > popmax] = popmax / 2  # 位置大于最大值  赋值为最大值的一半
        pop[pop < popmin] = popmin / 2  # 位置大于最小值 赋值为最小值的一半

        # 自适应变异
        p = np.random.random()  # 随机生成一个0~1内的数
        if p > 0.8:  # 如果这个数落在变异概率区间内，则进行变异处理
            k = np.random.randint(0, 2)  # 在[0,2)之间随机选一个整数
            pop[:, k] = np.random.random()  # 在选定的位置进行变异

        # 计算适应度值
        fitness = getFitness(X_train, X_test, y_train, y_test, pop)

        # 个体最优位置更新
        if fitness < fitnessgbest:
            index = np.argmin(fitness)
            fitnessgbest[index] = fitness[index]  # 获取最优适应度
            gbest[index] = pop[index]  # 获取最优位置

        # 群体最优更新
        i = np.argmin(fitness)  # 获取适应度最小值索引
        if fitness[i] < fitnesszbest:  # 适应度判断
            zbest = pop[i]  # 最优位置赋值
            fitnesszbest = fitness[i]  # 最优适用度赋值

        # 记录历史最优状态
        gbest_history.append(gbest)
        zbest_history.append(zbest)
        record_RODDPSO[t] = fitnesszbest  # 记录群体最优适应度的变化

        t = t + 1  # 迭代次数递增
    # 绘制适应度折线图
    plt.figure()  # 设置图片
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt.plot(record_RODDPSO, label='RODDPSO')  # 绘制折线图
    plt.xlabel('Iteration times')  #
    plt.ylabel('Fitness')
    plt.title('Fitness curve')
    plt.legend()  # 设置图例
    plt.show()  # 显示图片


    if abs(zbest[0][0]) > 0:  # 判断
        best_max_features = int(abs(zbest[0][0])) + 6  # 赋值
    else:
        best_max_features = int(abs(zbest[0][0])) + 8  # 赋值

    if abs(zbest[1][0]) > 0:  # 判断
        best_n_estimators = int(abs(zbest[1][0])) + 650  # 赋值
    else:
        best_n_estimators = int(abs(zbest[1][0])) + 1000  # 赋值

    print('----------------RODDPSO optimal results-----------------')
    print("The best max_features is " + str(abs(best_max_features)))
    print("The best n_estimators is " + str(abs(best_n_estimators)))

    print('----------------RODDPSO-RF Regression Model Evaluation-----------------')
    # 应用优化后的最优参数值构建随机森林回归模型
    rfc_model = RandomForestRegressor(max_features=best_max_features, n_estimators=best_n_estimators)  # 建模
    rfc_model.fit(X_train, y_train)  # 拟合
    y_pred = rfc_model.predict(X_test)  # 预测
    X_train_pred =rfc_model.predict(X_train)
    # 应用10折交叉验证 计算模型得分
    accuracies = cross_val_score(rfc_model, X=X_train, y=y_train, cv=10)
    accuracy_mean = accuracies.mean()  # 取10次交叉验证均值
    # 模型评估
    print('**************************Model performance on test dataset*******************************')
    print('RF Regressor Train R2: ', evaluate_model(rfc_model, X_train, y_train))
    print('RF Regressor Test R2: ', evaluate_model(rfc_model, X_test, y_test))
    print('RF Regressor Train R2: {}'.format(round(r2_score(y_train, X_train_pred),4)))
    print('RF Regressor Test R2: {}'.format(round(r2_score(y_test, y_pred), 4)))
    print('RF Regressor MSE: {}'.format(round(mean_squared_error(y_test, y_pred), 4)))
    print('RF Regressor EV: {}'.format(round(explained_variance_score(y_test, y_pred), 4)))
    print('RF Regressor MAE: {}'.format(round(mean_absolute_error(y_test, y_pred), 4)))

    # 真实值与预测值比对图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.plot(range(len(y_test)), y_test, color="blue", linewidth=1.5, linestyle="-")
    plt.plot(range(len(y_pred)), y_pred, color="red", linewidth=1.5, linestyle="-.")
    plt.legend(['Real', 'Prediction'])
    plt.title("RODDPSO-RF comparison between real and prediction")
    plt.show()  # 显示图片


