# Algorithm 1: Adaptive Hybrid Multi-Task GBDT for CVR Prediction

**Input:**
- 学習データ: D = {(x_i, y_{i,ctr}, y_{i,cvr}, y_{i,ctcvr})}_{i=1}^N
- ハイパーパラメータ:
    - 木の総数: M
    - 学習率: η
    - 正則化項: λ, γ
    - アンサンブル重み: w_ctr, w_cvr, w_ctcvr (デフォルト: 10, 1, 10)
    - ローカル戦略の閾値: Threshold_prop_ctcvr, Threshold_prop_cvr
    - OOF用の分割数: K

**Output:**
- 学習済みアンサンブルモデル: F_M

---

// フェーズ1：事前準備
pCTR_oof ← GenerateOOFCtrPredictions(D, K)

// フェーズ2：GBDTモデルの学習
// 1. 初期化
for each task j in {ctr, cvr, ctcvr} do
    F_{j,0}(x_i) ← InitialPrediction(y_j)
end for

// 2. 木の構築ループ
for m = 1 to M do
    // A. サンプルごとの生の勾配・ヘシアンを計算
    for each sample i = 1 to N do
        for each task j in {ctr, cvr, ctcvr} do
            if task j is applicable for sample i then // CVRは y_{i,ctr}=1 のみ
                p_{ij} ← sigmoid(F_{j, m-1}(x_i))
                g_{ij,raw} ← p_{ij} - y_{ij}
                h_{ij,raw} ← p_{ij} * (1 - p_{ij})
            end if
        end for
    end for

    // B. 正規化
    for each task j in {ctr, cvr, ctcvr} do
        G_{j,raw} ← all applicable raw gradients for task j
        g'_{j} ← Normalize(G_{j,raw}) // 平均・標準偏差を目標値に変換
        h'_{j} ← Normalize(H_{j,raw}) // ヘシアンも同様
    end for

// C. CVRのIPS補正
for each sample i where y_{i,ctr}=1 do
    p_{i,ctr_oof} ← pCTR_oof[i] // フェーズ1で計算したOOF予測値を使用
    g''_{i,cvr} ← g'_{i,cvr} * (1 / p_{i,ctr_oof})
    h''_{i,cvr} ← h'_{i,cvr} * (1 / p_{i,ctr_oof})
end for    // D. 新しい決定木 T_m の構築
    T_m ← BuildTree(root_node, {g', h', g'', h''})

    // E. モデルの更新
    for each task j in {ctr, cvr, ctcvr} do
        F_{j,m}(x) ← F_{j,m-1}(x) + η * T_{j,m}(x)
    end for
end for

return F_M

---
### Sub-routine: BuildTree

**Input:**
- Node: 分割対象のノード（サンプルインデックスの集合）
- Gradients & Hessians: {g', h', g'', h''}

**Output:**
- Tree: Nodeから成長させた木（または葉）
---

// 停止条件の確認
if Node meets stopping criteria (e.g., depth == max_depth) then
    return CreateLeaf(Node)
end if

// ローカル戦略：信号の信頼性に基づく動的フォールバック
prop_neg_ctcvr ← proportion of samples in Node where g_{i,ctcvr,raw} < 0
prop_neg_cvr ← proportion of samples in Node where g_{i,cvr,raw} < 0

// アンサンブル勾配・ヘシアンの計算
for each sample i in Node do
    if prop_neg_ctcvr >= Threshold_prop_ctcvr and prop_neg_cvr >= Threshold_prop_cvr then // 3タスクでアンサンブル
        g_{ie} ← w_{ctr}g'_{i,ctr} + w_{cvr}g''_{i,cvr} + w_{ctcvr}g'_{i,ctcvr}
        h_{ie} ← w_{ctr}h'_{i,ctr} + w_{cvr}h''_{i,cvr} + w_{ctcvr}h'_{i,ctcvr}
    else if prop_neg_ctcvr < Threshold_prop_ctcvr and prop_neg_cvr >= Threshold_prop_cvr // CTCVRを除外し、2タスクでアンサンブル
        g_{ie} ← w_{ctr}g'_{i,ctr} + w_{cvr}g''_{i,cvr}
        h_{ie} ← w_{ctr}h'_{i,ctr} + w_{cvr}h''_{i,cvr}
    else if prop_neg_ctcvr >= Threshold_prop_ctcvr and prop_neg_cvr < Threshold_prop_cvr // CVRを除外し、2タスクでアンサンブル
        g_{ie} ← w_{ctr}g'_{i,ctr} + w_{ctcvr}g'_{i,ctcvr}
        h_{ie} ← w_{ctr}h'_{i,ctr} + w_{ctcvr}h'_{i,ctcvr}
    else // CVR, CTCVRを除外し，CTRタスクのみ使用
        g_{ie} ← w_{ctr}g'_{i,ctr}
        h_{ie} ← w_{ctr}h'_{i,ctr}
    end if
end for

// 最適分岐の探索
Find best split θ* by maximizing Gain using {g_e, h_e} for samples in Node.

// 分岐の実行
if Gain(θ*) > γ then
    Split Node into LeftChild, RightChild based on θ*
    return InternalNode(split=θ*, left=BuildTree(LeftChild), right=BuildTree(RightChild))
else
    return CreateLeaf(Node)
end if

---
### Sub-routine: CreateLeaf

**Input:**
- Node: 葉となるノード（サンプルインデックスの集合）

**Output:**
- Leaf: 各タスクの出力値を持つ葉ノード
---

Initialize Leaf.outputs as an empty map.
for each task j in {ctr, cvr, ctcvr} do
    G_{j,raw} ← sum(g_{ij,raw} for i in Node)
    H_{j,raw} ← sum(h_{ij,raw} for i in Node)
    Leaf.outputs[j] ← - G_{j,raw} / (H_{j,raw} + λ)
end for
return Leaf

---
### Sub-routine: GenerateOOFCtrPredictions

**Input:**
- D: 学習データセット
- K: 分割数

**Output:**
- pCTR_oof: アウトオブフォールドCTR予測値のリスト
---

Initialize pCTR_oof as an array of size N.
Split indices {1,...,N} into K folds: F_1, ..., F_K.
for k = 1 to K do
    Train_indices ← All indices except F_k
    Val_indices ← F_k
    CTR_Model_k ← Train a single-task GBDT on D[Train_indices] for CTR.
    p_k ← CTR_Model_k.predict(D[Val_indices])
    pCTR_oof[Val_indices] ← p_k
end for
return pCTR_oof