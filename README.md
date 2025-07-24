# 做市商端到端收益预测项目架构（2025-07-20）

## 一、数据准备阶段

### 1. 数据获取
老师提供多来源数据：

- 股指期货（IF、IH、IC）：仅有 orderbook。  
- ETF / 股票：orderbook、逐笔成交、挂单撤单。
- 每种数据类型由统一接口接入，根据用户指定标的，自动调整预处理参数与模型输入。

参考文献：  
<https://centaur.reading.ac.uk/104707/1/mathematics-10-01234-v2.pdf>

### 2. 数据清理与预处理

#### 期货数据（Orderbook）
- 去极值：极值阈值设定为 3.5×1.4826×MAD ≈ ±3.5σ，仅剔除 ~0.06% 极端点
- Z-score 标准化：便于跨标的训练和比较和方便深度学习模型拟合
-  前 100 tick设置为none，tick≥100 时改用滚动窗口（100）进行上述操作
-  所有盘口价格一起进行上述操作
-  500 ms 时间轴对齐 → 缺失值向后填充
- 筛选交易时段（09:30–15:00），剔除集合竞价与隔夜段
- 可将隔夜收益（收盘价(t-1) → 次日开盘价(t)）作为特征输入


#### ETF / 股票数据
- 逐笔成交、挂单撤单数据无需处理  
- orderbook 同上处理  
- 统一字段与格式，保证跨标的兼容性  

---

## 二、双路线并行建模

### 路线一：端到端深度模型（全序列建模）
参考文献：  
<https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/blob/master/jupyter_pytorch/run_train_pytorch.ipynb>

#### 输入
- 原始多档 orderbook  
- 成交量序列  
- 挂单撤单等辅助数据
- 使用多元自回归模型（VAR）评估训练集数据的时间依赖结构，通过 AIC 曲线或遍历不同窗口长度作为回看窗口长度
- 参考文献：  https://ajcutuli.github.io/OrderBookDeepLearning/

#### 模型
- 基础：DeepLOB  
- 后续优化：CNN + LSTM / Transformer 组合  

#### 标签设计
- 标签向量：Y(t) = [label_buy, label_sell]  
- 预测目标  
  - label_buy：当前 bid1 与未来 ask1 比较  
  - label_sell：当前 ask1 与未来 bid1 比较  

#### 三分类规则（阈值 ε）


- 若市场波动性下降，模型大量输出中性信号（0），可考虑 动态调整 ε 阈值
  
| 标签 | 条件 | 含义 |
|---|---|---|
| 1 | ask1(t) – bid1(t+k) > ε | 盈利 |
| -1 | bid1(t+k) – ask1(t) > ε | 亏损 |
| 0 |abs(ask1(t) – bid1(t+k)) ≤ ε | 持平 |



#### 多周期分类
- 30 s / 1 min / 5 min  
- 参考文献：<https://arxiv.org/html/2403.09267v3>

#### 损失函数
- Loss_buy   = CrossEntropy(pred_buy_logits, label_buy)
- Loss_sell  = CrossEntropy(pred_sell_logits, label_sell)
- Total_Loss = Loss_buy + Loss_sell


---

### 路线二：gplearn 因子挖掘 + 回归模型
参考文献：  
<https://blog.csdn.net/qq_24099909/article/details/109292031>

#### 特征生成
1. 手动计算标准盘口特征  
   - bid/ask 价格、spread、mid_price、OFI、市场深度、成交量等  
2. 自动化因子  
   - 输入：基础盘口特征时间序列  
   - 算子：moving average、rolling std、log、delta、rank …  
   - 输出：因子公式树  

#### 运算优化
- 运算图优化  
- 限制公式深度  
- 多核并行回测  

#### 因子筛选
- 评价指标：RankIC（Spearman）  
- 标签：mid price（后续可调整）  
- 保留高 |RankIC| 且多周期稳定的因子  

#### 回归建模
- 模型：XGBoost / LightGBM  
- 预测：多周期涨跌幅  
- 损失函数：与路线一保持一致  

---

### 路线三：TLOB 类端到端 Transformer（可选）
参考文献：  
<https://2048.csdn.net/682edabb606a8318e859c5e0.html>

> 说明：TLOB 与路线一思路相同，仅在计算资源不足或 CNN+LSTM 效果不佳时作为前沿探索性备选；现阶段核心仍为路线一与路线二。

#### 输入
同路线一

#### 架构
- 时间轴自注意力（Time Attention）  
- 空间轴自注意力（Spatial Attention）  
- 双重注意力 + MLP/卷积融合  
- 轻量化注意力、残差连接、归一化  

#### 标签与训练
同路线一（双任务分类损失）

---

## 三、统一回测与评估体系

### 1. 回测指标
- 策略收益率
- 夏普  
- 胜率  
- 最大回撤
- 等其他指标

### 2. 回测框架
- 同时支持端到端模型 & 因子模型输出  

### 3. 多周期统一评估
- 30 s / 1 min / 5 min  
- 对比三条路线的策略效果与稳定性
