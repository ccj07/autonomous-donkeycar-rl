# Autonomous Driving Agent using Deep Reinforcement Learning (DQN / Double DQN / Dueling DQN)

本專案基於 **Gym DonkeyCar** 模擬器，使用 **PyTorch** 實作並評估三種深度強化學習演算法（Deep Q-Network, Double DQN, Dueling DQN）在端到端（End-to-End）自駕車循跡任務中的表現。

Agent 直接以車載攝影機影像作為輸入，經灰階轉換、正規化與連續 4 幀堆疊處理後，輸出車輛轉向與油門指令。

---

## 專案亮點

- 端到端視覺控制：僅依賴車前相機輸入，無使用人工標註路徑點或幾何感測器特徵。
- 三種演算法橫向對比：完整實作 Nature DQN、Double DQN、Dueling DQN，在相同環境與超參數下公平對比。
- 高維度動作空間離散化：轉向切分 20 階、油門切分 20 階，建構 400 種精細控制動作。
- Checkpoint 評估：訓練 20,000 Iterations，每 500 代保存權重，以 Greedy 策略（$\epsilon=0$）進行多回合評估並導出統計數據。

---

## 演算法機制與架構

| 演算法 | 特徵擷取層 | 決策/輸出層結構 | 目標值更新公式 |
| :--- | :--- | :--- | :--- |
| **DQN** | 2 層 Conv2D + ReLU | FC(512) $\to$ FC(400) | $r + \gamma \max_{a'} Q_{\text{target}}(s', a')$ |
| **Double DQN** | 2 層 Conv2D + ReLU | FC(512) $\to$ FC(400) | $r + \gamma Q_{\text{target}}(s', \arg\max_{a'} Q_{\text{predict}}(s', a'))$ |
| **Dueling DQN** | 2 層 Conv2D + ReLU | 分支為 $V(s)$ 與 $A(s, a)$ | $Q = V(s) + \left(A(s, a) - \frac{1}{\vert{}A\vert{}}\sum_{a'} A(s, a')\right)$ |

---

## 實驗結果總結

在 20,000 回合訓練中，每隔 500 回合保存模型權重，共測試 40 個 Checkpoint（每組跑 5 回合 Greedy Policy 取平均）：

| 指標 | 傳統 DQN | Dueling DQN | Double DQN |
| :--- | :---: | :---: | :---: |
| 全週期平均得分 (Overall Mean) | 29.35 | 53.60 | **117.99** |
| 最高平均得分 (Peak Mean) | 133.81 (Ep 18,500) | 148.33 (Ep 18,000) | **350.37 (Ep 19,500)** |
| 全週期平均存活步數 | 11.85 步 | 14.67 步 | **24.20 步** |
| 最高平均存活步數 | 25.8 步 | 28.0 步 | **60.8 步** |
| 單回合最高得分 | 208.99 | **402.38** | 381.39 |

> **核心結論**：面對 $20 \times 20 = 400$ 種離散動作的高維動作空間，傳統 DQN 與 Dueling DQN 易因 Max 操作產生嚴重的 Q 值高估問題；**Double DQN 透過動作選擇與評估解耦，展現了最快的收斂速度與最穩定的行駛表現（最高平均分達 350.37，存活步數達 60.8 步）**。

---

## 檔案結構

```text
├── DQN/
│   ├── donkeycar.ipynb   # DQN 訓練流程
│   ├── DQN_result.ipynb  # Checkpoint 批次評估腳本
│   ├── dqn_checkpoint_evaluation.csv  # DQN 測試數據報表
│   └── dqn_evaluation_curve.png       # DQN 數據曲線圖
├── Double_DQN/
│   ├── donkeycar_doubleDQN.ipynb  # Double_DQN 訓練流程
│   ├── double_result.ipynb        # Checkpoint 批次評估腳本
│   ├── double_dqn_checkpoint_evaluation.csv  # Double_DQN 測試數據報表
│   └── double_dqn_evaluation_curve.png       # Double_DQN 數據曲線圖
├── Dueling_DQN/
│   ├── donkeycar_duelingDQN.ipynb  # Dueling_DQN 訓練流程
│   ├── dueling_result.ipynb        # Checkpoint 批次評估腳本
│   ├── dueling_dqn_checkpoint_evaluation.csv  # Dueling_DQN 測試數據報表
│   └── dueling_dqn_evaluation_curve.png       # Dueling_DQN 數據曲線圖
└── README.md  # 專案說明文件
```

---
## AI 揭露
自動化評估架構與批次處理邏輯主要由 Google Gemini 生成

